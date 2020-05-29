
p = genParams();
%%
for curSet = 1:size(p.fixationSets, 1)
	curSetName = p.fixationSets{curSet};
	load(sprintf('%s/fixations/%sFix.mat', p.datasetDir, curSetName));
	load(sprintf('%s/fixations/%sSize.mat', p.datasetDir, curSetName));		% Benchmark faster by skipping reading image size info from jpg files.
	load(sprintf('%s/fixations/%sFixmap.mat', p.datasetDir, curSetName));
    imgNum = size(fixCell, 1);
    
    if ~exist(sprintf('%s/fixations/%sFixmap.mat', p.datasetDir, curSetName), 'file')
			if p.verbose
				fprintf('No fixation info available: %s\n', curSetName);
			end
			continue;
    end
    
	for curAlgNum = 1:size(p.fixationAlgs, 1)
		curAlgName = p.fixationAlgs{curAlgNum};
		outFileName = sprintf('%s/igdg/%s_%s.mat', p.outputDir, curSetName, curAlgName);
		if exist(outFileName, 'file') && ~p.reCal 
			if p.verbose
				fprintf('Skipping existing file: %s\n', outFileName);
			end
			continue;
        end
		tic
		allScore= zeros(1, imgNum);
       
        %% Cannot generate a good base fixation map since the size of the images are not identical.
        
		for curImgNum = 1:imgNum
%             fprintf('%s - %d\n',curSetName, curImgNum);
            fixationMap = fixmapCell{curImgNum};
            if strcmp(curAlgName,'hum')
                rawSMap = antonioGaussian(fixationMap, 8);
                rawSMap = rawSMap / max(rawSMap(:));
            else
                rawSMap = im2double(imread(sprintf('%s/%s/%s/%d.png', p.algMapDir, curSetName, curAlgName, curImgNum)));
                rawSMap = imresize(rawSMap, sizeData(curImgNum,:), 'bilinear');
            end
            baseMap = im2double(imread(sprintf('%s/%s/%s/%d.png', p.algMapDir, curSetName, 'dg2', curImgNum)));
            baseMap = baseMap/max(baseMap(:)); 
            score = InfoGain(rawSMap, fixationMap, imresize(baseMap,size(fixationMap)));
            allScore(1,curImgNum) = score; 
		end
		curTime = toc;

		if p.verbose
			fprintf('%s on %s done in %.2f seconds! Score = %f\n', curAlgName, curSetName, curTime, mean(allScore(:)));
        end
		% save result
		save(outFileName, 'allScore');
    end
end

