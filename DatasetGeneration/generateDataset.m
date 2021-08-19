function generateDataset(imsPerClass, currAngGenerator, scalingToChoose, targetDirectory)
    arguments
        imsPerClass (1,1) double {mustBeInteger,mustBePositive} = 1000
        currAngGenerator (1,1) function_handle = @()2*pi*rand()
        scalingToChoose (1,1) double = 1
        targetDirectory char = fullfile('..','Dataset')
    end
    % Get all valid folders
    directoryContent = dir('.');
    isValidDir = isfolder({directoryContent.name}) & cellfun(@strlength,{directoryContent.name})>2;
    folders = directoryContent(isValidDir);
    
    onlyConsider = {'arrow','semicircle','triangle','doublearrow','pentangon'}; % All 5 classes
    vals = cellfun(@(currStr)strcmp({folders.name},currStr),onlyConsider,'UniformOutput',false);
    folders = folders(any(cat(1,vals{:})));

    fns = {};
    angs = {};
    noOfSyms = {};
    % Iterate over all
    for dirNo = 1:numel(folders)
        currFolderName = fullfile(folders(dirNo).name,[num2str(scalingToChoose),'x']);
        allFiles = dir(currFolderName);
        assert(numel(allFiles)>2,'%s must not be empty',currFolderName);
        % Find shortest file name that is not . or ..
        allNameLengths = cellfun(@strlength,{allFiles.name});
        [~, ind] = min(allNameLengths+1e5*(allNameLengths<=2));
        imUnrotated = imread(fullfile(allFiles(ind).folder,allFiles(ind).name),'png','BackgroundColor',[1,1,1]);
        imUnrotated = imUnrotated(:,:,1); % Only use 1 channel to convert to greyscale
        assert(isequal(size(imUnrotated),[24,24]*scalingToChoose));
        if ~exist(targetDirectory,'dir')
            warning('Directory does not exist, creating...');
            mkdir(targetDirectory);
        end
        for i=1:imsPerClass
            % 255-... is dirty hack to ensure new pixels due to rotation are white
            % and not black
            currAng = currAngGenerator();
            % Takes degrees! Tested thoroughly
            imRotatedCurr = 255-imrotate(255-imUnrotated, rad2deg(currAng), 'bicubic', 'crop');
            %imwrite(imRoratedCurr, fullfile(allFiles(ind).folder,'..','aug',[allFiles(ind).name(1:end-3),num2str(i,'%03d'),'.png']));
            currName = [allFiles(ind).name(1:end-3),num2str(i,'%05d'),'.png'];
            fns = [fns,{currName}]; %#ok<AGROW>
            angs = [angs, {currAng}]; %#ok<AGROW>
            noOfSyms = [noOfSyms, {nameToSym(currName)}]; %#ok<AGROW>
            imwrite(imRotatedCurr, fullfile(targetDirectory,currName));
        end
    end

    dataLabelStruct = struct('filenames',fns, 'initial_angle',angs,'n_symmetries',noOfSyms);
    dataLabelTable = struct2table(dataLabelStruct);
    writetable(dataLabelTable, fullfile(targetDirectory,'orientationsWithSymm.csv'));

    function label=nameToSym(name)
        switch name(1:end-10)
            case 'circle'
                label = 0;
            case {'arrow', 'semicircle', 'semicircleInHexagon','hexacircle'}
                label = 1;
            case {'rectangle','doublearrow'}
                label = 2;
            case 'triangle'
                label = 3;
            case {'square', 'tetrastar'}
                label = 4;
            case {'pentagon', 'pentastar'}
                label = 5;
            case {'hexagon','hexastar'}
                label = 6;
            otherwise
                error('Class not recognized');
        end
    end
end