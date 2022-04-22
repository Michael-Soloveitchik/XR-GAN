import torch

from Models.Losses.losses import IoULoss, TverskyLoss, DiceLoss, FocalLoss
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_BCE', type=float, default=10.0, help='weight for BCE loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_cross_entropy_mean', 'G_cross_entropy_sum', 'G_Tversky', 'G_Dice', 'G_Jacard','G_focal_Tversky', 'G_focal_loss',\
                           'D_real', 'D_fake']
                          # ["G_L1_+"clss for clss in self.opt.output_classes]
        self.classes = opt.output_classes.split('_')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names_train = ['real_A'] + ['fake_B_'+clss for clss in self.classes] + ['real_B_'+clss for clss in self.classes]
        self.visual_names_test = ['real_test_A'] + ['fake_test_B_'+clss for clss in self.classes]
        self.visual_names = self.visual_names_train + self.visual_names_test

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        fraction = None if (opt.fraction_r <= 0 or opt.fraction_s <= 0) else (opt.fraction_r, opt.fraction_s)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, fraction=fraction)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, fraction=fraction)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterion_SCE = torch.nn.BCEWithLogitsLoss(reduction='sum')
            self.criterion_MCE = torch.nn.BCEWithLogitsLoss(reduction='mean')
            self.criterion_Jacard = IoULoss()
            self.criterion_Dice = DiceLoss()
            self.criterion_Tversky = TverskyLoss()
            self.criterion_focal_loss = FocalLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_test_A = input['test_A'].to(self.device).detach()
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        for clss in self.classes:
            self.__dict__['real_B_' + clss] = input['B_'+clss if AtoB else 'A_'+clss].to(self.device)[:self.opt.input_nc]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, train=True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if train:
            self.fake_B = self.netG(self.real_A)
            for i, clss in enumerate(self.classes):
                self.__dict__['fake_B_'+clss] = self.normalize_tensor(self.fake_B[:self.opt.input_nc, i:i+self.opt.input_nc])  # G(A)
        if not train:
            # self.fake_test_B = torch.repeat_interleave(self.netG(self.real_test_A[0][None,...]).detach(), self.real_test_A.shape[0]*0+1, axis=0)  # G(A)
            test = self.real_test_A[0][None, ...]
            test = test.to('cpu')
            net = self.netG.module.to('cpu')
            self.fake_test_B = net(test)  # G(A)
            for i, clss in enumerate(self.classes):
                self.__dict__['fake_test_B_'+clss] = self.normalize_tensor(self.fake_test_B[:self.opt.input_nc,i:i+self.opt.input_nc])
            self.netG.cuda()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.normalize_tensor(self.fake_B) ), 1)# we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def normalize_tensor(self,tensor):
        return (torch.nn.functional.sigmoid(tensor)-0.5)*2.
        # return torch.cat((tensor, ((-1./6.)*tensor.sum(1)[:,None,...]).to(tensor.device)), 1)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # for i, clss in enumerate(self.classes):
        fake_AB = torch.cat((self.real_A, self.normalize_tensor(self.fake_B)), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.normalize_tensor(self.fake_B) , self.real_B) * self.opt.lambda_L1
        # fake_B = self.self.normalize_tensor(self.fake_B) #torch.cat((self.fake_B, torch.ones((1, 1, 600, 600)).to(self.fake_B.device)), 1)
        # real_B = self.self.normalize_tensor(self.real_B) #torch.cat((self.real_B, torch.ones((1, 1, 600, 600)).to(self.real_B.device)*), 1)
        self.loss_G_cross_entropy_sum = self.criterion_SCE(self.fake_B, (self.real_B+1.0)/2.) /float(self.opt.batch_size)* self.opt.lambda_BCE
        self.loss_G_cross_entropy_mean = self.criterion_MCE(self.fake_B, (self.real_B+1.0)/2.) /float(self.opt.batch_size)* self.opt.lambda_BCE
        self.loss_G_Tversky = self.criterion_Tversky(self.fake_B, (self.real_B+1.0)/2.) /float(self.opt.batch_size)* self.opt.lambda_BCE
        self.loss_G_focal_Tversky = self.loss_G_Tversky**1.25
        self.loss_G_focal_loss = 0. #self.criterion_focal_loss(self.fake_B, (self.real_B+1.0)/2.) /float(self.opt.batch_size)* self.opt.lambda_BCE
        self.loss_G_Jacard = self.criterion_Jacard(self.fake_B, (self.real_B+1.0)/2.) /float(self.opt.batch_size)* self.opt.lambda_BCE
        self.loss_G_Dice = self.criterion_Dice(self.fake_B, (self.real_B+1.0)/2.) /float(self.opt.batch_size)* self.opt.lambda_BCE

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN +self.loss_G_focal_Tversky*1000# + self.loss_G_L1
        self.loss_G.backward()
    # def iou_bce(self,logits, target):
    #
    #     torch.nn.softmax(logits)
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()


        torch.cuda.empty_cache()
        # udpate G's weights
