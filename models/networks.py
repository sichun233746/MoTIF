import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.Ours as Ours
import models.modules.Ours_7 as Ours_7
import models.modules.Ours_4 as Ours_4
import models.modules.Ours_44 as Ours_44
import models.modules.Ours_flow as Ours_flow
import models.modules.Ours_ZSM as Ours_ZSM
import models.modules.ZSM as ZSM
import models.modules.Super_SloMo as Super_SloMo
import models.modules.EDVR as EDVR
import models.modules.Ours_back as Ours_back
import models.modules.TMNet as TMNet
####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LIIF':
        netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])   
    elif which_model == 'LunaTokis':
        netG = Sakuya_arch_o.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])  
    elif which_model == 'Ours':
        if 'setting' in opt_net:
            netG = Ours.LunaTokis(setting = opt_net['setting'])
        else:
            netG = Ours.LunaTokis()
    elif which_model == 'Ours_ZSM':
        netG = Ours_ZSM.LunaTokis(setting = opt_net['setting'])
    elif which_model == 'Ours_back':
        netG = Ours_back.LunaTokis(setting = 5)
    elif which_model == 'Ours_7':
        netG = Ours_7.LunaTokis(setting = opt_net['setting'])
    elif which_model == 'Ours_4':
        netG = Ours_4.LunaTokis() 
    elif which_model == 'Ours_44':
        netG = Ours_44.LunaTokis() 
    elif which_model == 'ZSM':
        netG = ZSM.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs']) 
    elif which_model == 'TMNet':
        netG = TMNet.TMNet(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs']) 
    elif which_model == 'Super_SloMo':
        netG = Super_SloMo.Net()  
    elif which_model == 'EDVR':
        netG = EDVR.EDVR() 
    elif which_model == 'Ours_flow':
        netG = Ours_flow.LunaTokis() 
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
