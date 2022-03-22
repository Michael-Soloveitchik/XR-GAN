from .base_model import BaseModel
from . import networks
import torch
from util.pyramids import apply_pyramid, normal2uint, uint2normal
from util.pyramids import apply_pyramid, normal2uint, uint2normal
from util.losses import TotalVariation, VGGPerceptualLoss
class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix, 'G' + opt.model_suffix+'_lp','G' + opt.model_suffix+'_hp']  # only generator is needed.
        self.netG =      networks.define_G(opt.input_nc*2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A_lp = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_lp, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A_hp = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_hp, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.apply_pyramid = lambda x: apply_pyramid(x, self.opt.pyramid_lavels) # ( #)lambda x: pyrUp(pyrDown(x, k=opt.pyramid_lavels), k=opt.pyramid_lavels)

    # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']
    def get_lowpass(self, im):
        return uint2normal(self.apply_pyramid(normal2uint(im)))

    def get_highpass(self, im):
        return uint2normal(normal2uint(im) - self.apply_pyramid(normal2uint(im)))
    def forward(self):
        """Run forward pass."""
        # self.fake = self.netG(self.real)  # G(real)
        self.fake_lp = self.netG_A_lp(self.get_lowpass(self.real[0][None].detach())).detach() # G_A(A)
        self.fake_hp = self.netG_A_hp(self.get_highpass(self.real[0][None].detach())).detach()  # G_A(A)
        self.fake =self.netG_A(torch.cat([self.fake_lp.detach(), self.fake_hp.detach()], dim=1)).detach()  # G_A(A)
    def optimize_parameters(self):
        """No optimization for test model."""
        pass
