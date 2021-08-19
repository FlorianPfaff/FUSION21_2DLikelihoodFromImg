% @author Florian Pfaff pfaff@kit.edu
% @date 2021
addpath('prepareFig')
%% Figure 2 Figure showing the trigonometric polynomial obtained via moment matching
figure(20);
wd = WDDistribution([1,1+pi],[0.5,0.5]);

col = colororder;
figure(21),clf,hold on

setupAxisCircular('x'),shg
fd11=FourierDistribution.fromDistribution(wd,11);
fd21=FourierDistribution.fromDistribution(wd,35);
fd51=FourierDistribution.fromDistribution(wd,81);

yyaxis right
handleWd = wd.plot(color=col(1,:));
yyaxis left
handleFd11 = fd11.plot(resolution=500,color=col(2,:),LineStyle='--');
handleFd21 = fd21.plot(resolution=500,color=col(3,:),LineStyle='-.');
handleFd51 = fd51.plot(resolution=500,color=[0.4660    0.6740    0.1880],LineStyle='-');
ylabel('Normalized likelihoood')

ax = gca();
ax.YAxis(1).Color=[0,0,0];
ax.YAxis(2).Color=col(1,:);
yyaxis right
ylabel('Sample weights')
ax.YAxis(2).Limits = [0,wd.w(1)];

legend([handleWd,handleFd11,handleFd21,handleFd51],{'Samples','11 coefficients','35 coefficients','81 coefficients'})
%% Save plot
prepareFig([9.8,5])
exportgraphics(gca,'wdMatching.pdf',ContentType='Vector')
%% Paths for Figures 3 to 5
figure(30);
filesArrow = dir('rotateArrow*');
filesSemicircle = dir('rotateSemicircle*');
filesDoublearrow = dir('rotateDoublearrow*');
filesTriangle = dir('rotateTriangle*');
filesPentagon = dir('rotatePentagon*');
% Use last file. Overwrite to set manually for reproducibility
fnArrow = fullfile(filesArrow(end).folder,filesArrow(end).name);
fnSemicircle = fullfile(filesSemicircle(end).folder,filesSemicircle(end).name);
fnDoublearrow = fullfile(filesDoublearrow(end).folder,filesDoublearrow(end).name);
fnTriangle = fullfile(filesTriangle(end).folder,filesTriangle(end).name);
fnPentagon = fullfile(filesPentagon(end).folder,filesPentagon(end).name);
%% Figure 3 showing the errors
% Get deviations
deviationsArrow = plotResultsBoxplot(fnArrow); 
deviationsSemicircle = plotResultsBoxplot(fnSemicircle);
deviationsDoublearrow = plotResultsBoxplot(fnDoublearrow);
deviationsTriangle = plotResultsBoxplot(fnTriangle);
deviationsPentangle = plotResultsBoxplot(fnPentagon);
% Repackage data for using boxplotFromCell
deviationsArrow = cellfun(@(cell){cell2mat(cell)},deviationsArrow);
deviationsSemicircle = cellfun(@(cell){cell2mat(cell)},deviationsSemicircle);
deviationsDoublearrow = cellfun(@(cell){cell2mat(cell)},deviationsDoublearrow);
deviationsTriangle = cellfun(@(cell){cell2mat(cell)},deviationsTriangle);
deviationsPentangle = cellfun(@(cell){cell2mat(cell)},deviationsPentangle);
%
boxplotFromCell([deviationsArrow,deviationsSemicircle,deviationsDoublearrow,deviationsTriangle,deviationsPentangle],...
    {'arrow','semicircle','double arrow','triangle','pentagon'})
ylabel('Symmetry-aware error in radian');
%% Fine-tune and save plot
ylim([-0.001,0.05])
prepareFig([10,5])
exportgraphics(gca,'errorBoxplot.pdf',ContentType='Vector')

%% Figure 4
expectedDeviaionBinnedCell = {};
actualDeviationBinnedCell = {};
figure(41);
[avgExpectedBinned, avgActualBinned] = showExpectedVSTrueDeviation(fnArrow);
expectedDeviaionBinnedCell = [expectedDeviaionBinnedCell,{avgExpectedBinned}];
actualDeviationBinnedCell = [actualDeviationBinnedCell, {avgActualBinned}];
figure(42);
[avgExpectedBinned, avgActualBinned] = showExpectedVSTrueDeviation(fnSemicircle);
expectedDeviaionBinnedCell = [expectedDeviaionBinnedCell,{avgExpectedBinned}];
actualDeviationBinnedCell = [actualDeviationBinnedCell, {avgActualBinned}];
figure(43);
[avgExpectedBinned, avgActualBinned] = showExpectedVSTrueDeviation(fnDoublearrow);
expectedDeviaionBinnedCell = [expectedDeviaionBinnedCell,{avgExpectedBinned}];
actualDeviationBinnedCell = [actualDeviationBinnedCell, {avgActualBinned}];
figure(44);
[avgExpectedBinned, avgActualBinned] = showExpectedVSTrueDeviation(fnTriangle);
expectedDeviaionBinnedCell = [expectedDeviaionBinnedCell,{avgExpectedBinned}];
actualDeviationBinnedCell = [actualDeviationBinnedCell, {avgActualBinned}];
figure(45);
[avgExpectedBinned, avgActualBinned] = showExpectedVSTrueDeviation(fnPentagon);
expectedDeviaionBinnedCell = [expectedDeviaionBinnedCell,{avgExpectedBinned}];
actualDeviationBinnedCell = [actualDeviationBinnedCell, {avgActualBinned}];
%% Fine-tune and save plot
figure(1045);
ylim([0,0.04])
xlim([0.025,0.085])
prepareFig([10,5])
exportgraphics(gca,'expectedVsTrueDeviationPentAll.pdf',ContentType='Vector')
%% Figure 5
figure(50),hold on
mSize = 3;
plot(expectedDeviaionBinnedCell{1},actualDeviationBinnedCell{1},'-o','MarkerSize',mSize)
plot(expectedDeviaionBinnedCell{2},actualDeviationBinnedCell{2},'--+','MarkerSize',mSize)
plot(expectedDeviaionBinnedCell{3},actualDeviationBinnedCell{3},'-.*','MarkerSize',mSize)
plot(expectedDeviaionBinnedCell{4},actualDeviationBinnedCell{4},':x','MarkerSize',mSize)
plot(expectedDeviaionBinnedCell{5},actualDeviationBinnedCell{5},'--d','MarkerSize',mSize)
legend({'Arrow','Semicirlce','Double arrow','Triangle','Pentagon'},'Location','Northwest')
xlim([0.025,0.085]);
xlabel('Expected error');
ylabel('Actual error');
%% Save
prepareFig([10,5])
exportgraphics(gca,'expectedVsTrueDeviationPentBinnedAllShapes.pdf',ContentType='Vector')