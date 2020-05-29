%   This script generates AUC scores for all fixationAlgs on all fixationSets.
%   Xiaodi Hou <xiaodi.hou@gmail.com>, 2014
%   Please email me if you find bugs or have questions.
p = genParams();
%%
for curSet = 1:size(p.fixationSets, 1)
	curSetName = p.fixationSets{curSet};
	load(sprintf('%s/fixations/%sFix.mat', p.datasetDir, curSetName));
	load(sprintf('%s/fixations/%sSize.mat', p.datasetDir, curSetName));		% Benchmark faster by skipping reading image size info from jpg files.
	load(sprintf('%s/fixations/%sFixmap.mat', p.datasetDir, curSetName));
   
    imgNum = size(fixCell, 1);
	for curAlgNum = 1:size(p.fixationAlgs, 1)
		curAlgName = p.fixationAlgs{curAlgNum};
		outFileName = sprintf('%s/auc/%s_%s.mat', p.outputDir, curSetName, curAlgName);
		if exist(outFileName, 'file') && ~p.reCal 
			if p.verbose
				fprintf('Skipping existing file: %s\n', outFileName);
			end
			continue;
		end

		tic
		allAUC = zeros(p.sigmaLen, imgNum);
		allNegFix = aucCore.genNegFix(fixCell, sizeData);
		
		for curImgNum = 1:imgNum
			% collect positive/negative fixations
			posFix = fixCell{curImgNum};
			
			negFixInd = 1:imgNum;
			negFixInd(curImgNum) = [];
			
			negFix = bsxfun(@plus, cell2mat(allNegFix(negFixInd)), round(sizeData(curImgNum, :)/2));
			inFixInd = negFix(:,1)>=1 & negFix(:,2)>=1 & negFix(:,1)<=sizeData(curImgNum,1) & negFix(:,2)<=sizeData(curImgNum,2);
            negFix = negFix(inFixInd, :);

      
			% generate full-sized saliency maps
			fixationMap = fixmapCell{curImgNum};
            if strcmp(curAlgName,'hum')
                rawSMap = antonioGaussian(fixationMap, 8);
                rawSMap = rawSMap / max(rawSMap(:));
            else
                rawSMap = im2double(imread(sprintf('%s/%s/%s/%d.png', p.algMapDir, curSetName, curAlgName, curImgNum)));
                rawSMap = imresize(rawSMap, sizeData(curImgNum,:), 'bilinear');
            end
			kSizeList = norm(sizeData(curImgNum,:)).*p.sigmaList;
			
			if p.useGPU
				tmpAUC = aucCore.benchImgGPU(rawSMap, posFix, negFix, kSizeList);
			else
				tmpAUC = aucCore.benchImg(rawSMap, posFix, negFix, kSizeList);
			end
			
			allAUC(:, curImgNum) = tmpAUC;
		end
		curTime = toc;	
		% save results
		allAUC = mean(allAUC, 2);
		sigmaList = p.sigmaList;
		save(outFileName, 'allAUC', 'sigmaList');
        
        if p.verbose
			fprintf('%s on %s done in %.2f seconds! Score = %f\n', curAlgName, curSetName, curTime, max(allAUC));
		end
	end
end

