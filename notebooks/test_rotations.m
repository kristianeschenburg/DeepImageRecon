addpath('/Users/kristianeschenburg/Documents/MATLAB/freesurfer/')
addpath('/Users/kristianeschenburg/Downloads/makeFslXfmMatrix/');

datapath = char('/Users/kristianeschenburg/Desktop/Research/Data/');
s0_file = sprintf('%sDiffusion/dtifit_S0.nii.gz',datapath);

%%
s0 = load_nifti(s0_file);

data = s0.vol;

xrad = -0.001431244375789239;
yrad = 0.0006047126224049633;
zrad = 0.002425133537365235;

xshift = -0.03860166454885112;
yshift = -0.024705442422709254;
zshift = -0.020878442652023342;

T = [xshift yshift zshift];
R = [xrad yrad zrad];
S = [1 1 1];

%%

M = makeFslXfmMatrix(T,R,S,'test_matrix.mat')

%%

tform = affine3d(R);

%%
reference = imref3d(size(data));
warped = imwaclearrp(data,tform,'OutputView',reference);