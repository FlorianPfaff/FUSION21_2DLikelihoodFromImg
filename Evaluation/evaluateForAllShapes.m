% @author Florian Pfaff pfaff@kit.edu
% @date 2021
addpath(genpath(fullfile('libDirectional','lib')));
addpath('FilterEvaluationFramework');
netPath = fullfile('..','PretrainedModel');
netName = 'symmetryDataset5CL81coFourierComplexEpoch30.ckpt';
pythonCodePath = fullfile('..','Training');

copyfile(fullfile(netPath,netName), netName);
scenarioNames={'rotateSemicircle','rotateArrow','rotateDoublearrow','rotatePentagon','rotateTriangle'};

rng(0);
for i=1:numel(scenarioNames)
    scenarioName=scenarioNames{i};
    noRuns=1000;
    % Increase if more coefficients are output. For low numbers of
    % coefficients, consider using more than the network outputs so only
    % little approximation errors are introduced by the filter
    filters=struct('name',{'sqff'},'filterParams',{81}); 

    netAndCodeParam = struct('netName',netName,'pythonCodePath',pythonCodePath);
    scenarioParams = customScenarioLikelihoodFromImg(scenarioName,netAndCodeParam);
    startEvaluation(scenarioName,filters,noRuns,scenarioCustomizationParams=scenarioParams);
end