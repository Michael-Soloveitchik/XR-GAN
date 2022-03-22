import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.pyramids import apply_pyramid, normal2uint, uint2normal
from util.losses import TotalVariation, VGGPerceptualLoss, ContentLoss
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').



    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether train0ing phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--gamma', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lp_gamma', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--hp_gamma', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'idt_A_L2', 'idt_G_A_perceptual', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'idt_B_L2', 'idt_G_B_perceptual', 'G_B_perceptual', 'G_A_perceptual']

        self.loss_names = self.loss_names# + self.loss_names_hp + self.loss_names_lp
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_test_A', 'fake_test_B', 'fake_test_B_hp','fake_test_B_lp']#, 'rec_test_A', 'rec_test_A_lp', 'rec_test_A_hp']
        visual_names_B = ['real_test_B', 'fake_test_A', 'fake_test_A_lp', 'fake_test_A_hp']#,'rec_test_B', 'rec_test_B_lp', 'rec_test_B_hp']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A_hp', 'G_B_hp', \
                                'G_A_lp', 'G_B_lp', \
                                'D_A', 'D_B',
                                'G_A', 'G_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A_hp', 'G_B_hp', \
                                'G_A_lp', 'G_B_lp',
                                'G_A', 'G_B']
        # self.pyrUp = get_pyrUp(size=opt., k=opt.pyramid_lavels)
        # self.pyrDown = get_pyrDown(size=opt., k=opt.pyramid_lavels)
        self.apply_pyramid = lambda x: apply_pyramid(x, self.opt.pyramid_lavels) # ( #)lambda x: pyrUp(pyrDown(x, k=opt.pyramid_lavels), k=opt.pyramid_lavels)
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc*2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc*2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A_lp = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_lp, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B_lp = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG_lp, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A_hp = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_hp, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B_hp = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG_hp, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_lp_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_hp_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_lp_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_hp_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # self.TV_loss = TotalVariation()
            self.perceptual_loss = VGGPerceptualLoss() #VGGPerceptualLoss()
            self.perceptual_loss.cuda()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netG_A_lp.parameters(), self.netG_B_lp.parameters(), self.netG_A_hp.parameters(), self.netG_B_hp.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def delete_losses_attributes(self):

        del self.loss_G
        del self.loss_G_A
        del self.loss_G_B
        del self.loss_cycle_A
        del self.loss_cycle_B
        del self.loss_idt_A
        del self.loss_idt_B
        del self.loss_idt_A_L2
        del self.loss_idt_B_L2
        del self.loss_idt_G_A_perceptual
        del self.loss_idt_G_B_perceptual

        del self.loss_D_B
        del self.loss_D_A

        del self.loss_G_A_perceptual
        del self.loss_G_B_perceptual

    def delete_test_attributes(self):
        del self.fake_test_B_lp
        del self.fake_test_A_lp
        del self.fake_test_B_hp
        del self.fake_test_A_hp
        del self.fake_test_B
        del self.fake_test_A

    def delete_attributes(self):
        del self.fake_B_lp
        del self.rec_A_lp
        del self.fake_A_lp
        del self.rec_B_lp
        del self.fake_B_hp
        del self.rec_A_hp
        del self.fake_A_hp
        del self.rec_B_hp
        del self.fake_B
        del self.rec_A
        del self.fake_A
        del self.rec_B


    def get_lowpass(self, im):
        return uint2normal(self.apply_pyramid(normal2uint(im)))

    def get_highpass(self, im):
        return uint2normal(normal2uint(im) - self.apply_pyramid(normal2uint(im)))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # if 'test_A' in input and 'test_B' in input:
        self.real_test_A = input['test_A' if AtoB else 'test_B'].to(self.device)
        self.real_test_B = input['test_B' if AtoB else 'test_A'].to(self.device)
        self.test_image_paths = input['test_A_paths' if AtoB else 'test_B_paths']

    def forward(self, test):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not test:

            self.fake_B_lp = self.netG_A_lp(self.get_lowpass(self.real_A)) # G_A(A)
            self.fake_A_lp = self.netG_B_lp(self.get_lowpass(self.real_B))  # G_B(B)


            self.fake_B_hp = self.netG_A_hp(self.get_highpass(self.real_A))  # G_A(A)
            self.fake_A_hp  = self.netG_B_hp(self.get_highpass(self.real_B))  # G_B(G_A(A))
            self.fake_B = self.netG_A(torch.cat([self.fake_B_lp,self.fake_B_hp], dim=1))  # G_A(A)
            self.fake_A = self.netG_B(torch.cat([self.fake_A_lp, self.fake_A_hp], dim=1))
            # G_A(G_B(B))

            self.rec_B_lp  = self.netG_A_lp(self.get_lowpass(self.fake_A)) # G_A(G_B(B))
            self.rec_A_lp  = self.netG_B_lp(self.get_lowpass(self.fake_B))  # G_B(G_A(A))
            self.rec_B_hp  = self.netG_A_hp(self.get_highpass(self.fake_A))
            self.rec_A_hp  = self.netG_B_hp(self.get_highpass(self.fake_B))  # G_B(G_A(A))
            self.rec_B  = self.netG_A(torch.cat([self.rec_B_lp, self.rec_B_hp], dim=1))    # G_A(G_B(B))
            self.rec_A  = self.netG_B(torch.cat([self.rec_A_lp, self.rec_A_hp], dim=1))  # G_B(G_A(A))
        else:
            torch.cuda.empty_cache()
            with torch.no_grad():
                self.fake_test_B_lp = self.netG_A_lp(self.get_lowpass(self.real_test_A[0][None].detach())).detach() # G_A(A)
                # self.rec_test_A_lp  = self.netG_B_lp(self.fake_test_B_lp.detach())  # G_B(G_A(A))
                self.fake_test_A_lp = self.netG_B_lp(self.get_lowpass(self.real_test_B[0][None].detach())).detach()  # G_B(B)
                # self.rec_test_B_lp  = self.netG_A_lp(self.fake_test_A_lp.detach())    # G_A(G_B(B))

                self.fake_test_B_hp = self.netG_A_hp(self.get_highpass(self.real_test_A[0][None].detach())).detach()  # G_A(A)
                # self.rec_test_A_hp  = self.netG_B_hp(self.fake_test_B_hp.detach())  # G_B(G_A(A))
                self.fake_test_A_hp = self.netG_B_hp(self.get_highpass(self.real_test_B[0][None].detach())).detach()  # G_B(B)
                # self.rec_test_B_hp  = self.netG_A_hp(self.fake_test_A_hp.detach()) # G_A(G_B(B))

                self.fake_test_B =self.netG_A(torch.cat([self.fake_test_B_lp.detach(), self.fake_test_B_hp.detach()], dim=1)).detach()  # G_A(A)
                # self.rec_test_A  = uint2normal(normal2uint(self.rec_test_A_lp.detach()) + normal2uint(self.rec_test_A_hp.detach()))  # G_B(G_A(A))
                self.fake_test_A = self.netG_B(torch.cat([self.fake_test_A_lp.detach(), self.fake_test_A_hp.detach()], dim=1)).detach()  # G_B(B)
                # self.rec_test_B  = uint2normal(normal2uint(self.rec_test_B_lp.detach()) + normal2uint(self.rec_test_B_hp.detach()))



    def backward_D_basic(self, netD, real, fake, gamma=1.):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5# * gamma
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""

        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, gamma=self.opt.gamma)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""


        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, gamma=self.opt.gamma)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A_lp = self.netG_A_lp(self.get_lowpass(self.real_B))
            # # # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B_lp = self.netG_B_lp(self.get_lowpass(self.real_A))
            # #
            # # # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A_hp = self.netG_A_hp(self.get_highpass(self.real_B))
            # # # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B_hp = self.netG_B_hp(self.get_highpass(self.real_A))

            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(torch.cat([self.idt_A_lp, self.idt_A_hp] ,dim=1))
            self.loss_idt_A_L2 = self.criterionIdt(self.idt_A, self.real_B)
            self.loss_idt_G_A_perceptual = 0#self.perceptual_loss(self.idt_A, self.real_B)#/100000
            self.loss_idt_A = (self.loss_idt_A_L2 + self.loss_idt_G_A_perceptual) * lambda_A * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(torch.cat([self.idt_B_lp, self.idt_B_hp] ,dim=1))
            self.loss_idt_B_L2 = self.criterionIdt(self.idt_B, self.real_A)
            self.loss_idt_G_B_perceptual = 0#self.perceptual_loss(self.idt_B, self.real_A)#/100000
            self.loss_idt_B = (self.loss_idt_B_L2 + self.loss_idt_G_B_perceptual) * lambda_B * lambda_idt

        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) #* lambda_B
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) #* lambda_A

        self.loss_G_B_perceptual = 0 #self.perceptual_loss(self.fake_A, self.real_A) * lambda_B
        self.loss_G_A_perceptual = 0 #self.perceptual_loss(self.fake_B, self.real_B) * lambda_A
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_B
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_A
        self.loss_G = (self.loss_G_A + \
                       self.loss_G_B + \
                       self.loss_cycle_A + \
                       self.loss_cycle_B + \
                       self.loss_idt_A + \
                       self.loss_idt_B + \
                       self.loss_G_A_perceptual + \
                       self.loss_G_B_perceptual) * self.opt.gamma
        # self.loss_G = self.loss_G + self.loss_G_lp + self.loss_G_hp
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(test=False)      # compute fake images and reconstruction images.
        torch.cuda.empty_cache()

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        torch.cuda.empty_cache()

        # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_A_lp, self.netD_A_hp, self.netD_B, self.netD_B_lp, self.netD_B_hp], True)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        torch.cuda.empty_cache()

