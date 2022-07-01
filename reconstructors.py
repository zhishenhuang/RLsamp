from header import *

def sigpy_solver(ksp,L=1e-4,max_iter=100,heg=192,wid=144,solver=None):
    '''
    sigpy solver for image reconstruction
    input shape:  [N,1,H,W]
    output shape: [N,1,H,W]
    '''
    mps = np.ones((1,heg,wid))
    x_recon = torch.zeros(ksp.shape)
    for ind in range(len(ksp)):
        x_recon[ind,0,:,:] = torch.tensor(np.fft.ifftshift(np.abs(TotalVariationRecon(ksp[ind,0:1,:,:].numpy(), mps, L,
                    max_iter=max_iter,show_pbar=False,solver=solver).run())) )
    return x_recon