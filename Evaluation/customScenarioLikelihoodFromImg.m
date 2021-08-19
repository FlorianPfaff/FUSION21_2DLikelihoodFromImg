function scenarioParam = customScenarioLikelihoodFromImg(scenario, scenarioCustomizationParams)
% @author Florian Pfaff pfaff@kit.edu
% @date 2021
pe = pyenv; % Need python for our file
if double(pe.Version)<3.8 || double(pe.Version)>3.9
    warning('Python version may not be supported');
end
pythonCodePath = scenarioCustomizationParams.pythonCodePath;
if count(py.sys.path,pythonCodePath) == 0
    insert(py.sys.path,int32(0),pythonCodePath);
end
scenarioParam.netName = scenarioCustomizationParams.netName;

assert(contains(scenarioParam.netName,'Fourier')||contains(scenarioParam.netName,'Grid'),'Write Fourier in checkpoint name to choose Fourier-based likelihood.')
if ~contains(scenarioParam.netName,'Fourier') % For likelihoods for grid filter
    py.importlib.import_module('OptimizedGridLikelihoodLayeredMassiveChange');
    scenarioParam.mlModel = py.OptimizedGridLikelihoodLayeredMassiveChange.LikelihoodModel.load_from_checkpoint(scenarioParam.netName);
else % For likelihoods for grid filter
    py.importlib.import_module('likelihood_from_img');
    numberOfCoeffs = int32(sscanf(scenarioParam.netName,'symmetryDataset5CL%dcoFourier*'));
    if contains(scenarioParam.netName,'Real')
        args = pyargs(use_real=true, n_coeffs = numberOfCoeffs);
    elseif contains(scenarioParam.netName,'Complex')
        args = pyargs(use_real=false, n_coeffs = numberOfCoeffs);
    else
        error('Unknown mode')
    end
    scenarioParam.mlModel = py.likelihood_from_img.LikelihoodModel.load_from_checkpoint(scenarioParam.netName,args);
end
scenarioParam.useTransition = false;
scenarioParam.useLikelihood = true;

shapeName = lower(sscanf(scenario,'rotate%s'));
scenarioParam.imPath = fullfile('..','DatasetGeneration',shapeName,'1x',[shapeName,'.png']);
assert(exist(scenarioParam.imPath,'file'),'Image file not found.');

switch shapeName
    case {'arrow','semicircle'}
        scenarioParam.manifoldType = 'circleSymm1MinExpDev';
    case 'doublearrow'
        scenarioParam.manifoldType = 'circleSymm2MinExpDev';
    case 'triangle'
        scenarioParam.manifoldType = 'circleSymm3MinExpDev';
    case 'pentagon'
        scenarioParam.manifoldType = 'circleSymm5MinExpDev';
    otherwise
        scenarioParam.imPath = scenarioCustomizationParams.imPath;
end
img = imread(scenarioParam.imPath,'png','BackgroundColor',[1,1,1]);
scenarioParam.dataset_mean = 239.3254;
scenarioParam.dataset_std = 50.4115;
scenarioParam.baseImg=img(:,:,1);
scenarioParam.measGenerator = ...
    @(x)reshape((double(255-imrotate(255-scenarioParam.baseImg, rad2deg(x), 'bicubic', 'crop'))...
    -scenarioParam.dataset_mean)/scenarioParam.dataset_std,[],1);
scenarioParam.timesteps = 10;
scenarioParam.measPerStep = 1;
scenarioParam.sysNoise = VMDistribution(0,100);
scenarioParam.initialPrior = CircularUniformDistribution();
scenarioParam.genNextStateWithoutNoise = @(x)x;
% Do not use grid values for likelihood because we directly
% provide them. Assume image is given as vector (for
% compatibility with framework
scenarioParam.useLikelihood = true;
if ~contains(scenarioParam.netName,'Fourier')
    scenarioParam.likelihood = @(z,x)...
        exp(single(scenarioParam.mlModel(...
            py.torch.tensor(py.numpy.array(...
                reshape(z,[1,size(scenarioParam.baseImg)])),pyargs(dtype=py.torch.float)))...
        .detach().numpy()))*double(scenarioParam.mlModel.n_gridpoints)/(2*pi);
else
    scenarioParam.likelihoodGenerator = @likelihoodGeneratorFourier;
end
function likelihood = likelihoodGeneratorFourier(z)
    outputData = single(scenarioParam.mlModel(...
            py.torch.tensor(py.numpy.array(...
                reshape(z,[1,size(scenarioParam.baseImg)])),pyargs(dtype=py.torch.float)))...
        .detach().numpy());
    if contains(scenarioParam.netName, 'Real')
        likelihood = FourierDistribution(outputData(1, 1:(size(outputData, 2) + 1) / 2), ...
            outputData(1, (size(outputData, 2) + 1) / 2 + 1:end), 'sqrt');
    elseif contains(scenarioParam.netName, 'Complex')
        a = 2 * outputData(1, 1:(size(outputData, 2) + 1)/2);
        b = -2 * outputData(1, (size(outputData, 2) + 1)/2+1:end);
        likelihood = FourierDistribution(a, b, 'sqrt');
    else
        error('Unknown mode.')
    end
end
end

