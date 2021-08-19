function [avgExpectedBinned, avgActualBinned] = showExpectedVSTrueDeviation(filename, numberOfBins)
arguments
    filename char {mustBeNonempty}
    numberOfBins (1,1) double = 20
end
load(filename,'scenarioParam','results','groundtruths')
mode=scenarioParam.manifoldType;

meanCalculationSymm = 'Bingham';
[distanceFunction,extractMean]=getDistanceFunMeanCalcAndLabel(mode);

[allDeviationsLastMat, allExpectedDeviationsLastMat] = ...
    determineAllDeviations(results, extractMean, distanceFunction, meanCalculationSymm, groundtruths);

figure(gcf().Number+1000);

expectedDeviationPosterior=allExpectedDeviationsLastMat{:};
actualDeviationPosterior=allDeviationsLastMat{:};

scatter(allExpectedDeviationsLastMat{:},allDeviationsLastMat{:},1.5)
xlabel('Expected error');
ylabel('Actual error');
%% Put into bins
expectedDeviationPosterior(1:10)=[]; % Discard first 10 because effect of prior is high there
actualDeviationPosterior(1:10)=[];
[expectedDeviationPosteriorSorted,order]=sort(expectedDeviationPosterior);
actualDeviationPosteriorSortedAccordingToExp = actualDeviationPosterior(order);

binIndexLimits = [1,(numel(actualDeviationPosterior)./numberOfBins)*(1:numberOfBins)];

avgExpectedBinned = NaN(1,numel(binIndexLimits)-1);
avgActualBinned = NaN(1,numel(binIndexLimits)-1);
for i = 1:numel(binIndexLimits)-1
    avgExpectedBinned(i) = mean(expectedDeviationPosteriorSorted(binIndexLimits(i):binIndexLimits(i+1)));
    avgActualBinned(i) = mean(actualDeviationPosteriorSortedAccordingToExp(binIndexLimits(i):binIndexLimits(i+1)));
end
figure(gcf().Number-1000);
plot(avgExpectedBinned,avgActualBinned)
xlabel('Expected error (average per bin)');
ylabel('Actual error (average per bin)');
end