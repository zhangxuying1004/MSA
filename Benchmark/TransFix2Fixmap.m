clear; clc;
p = genParamsSCAFITIP17();
%%
for curSet = 1:size(p.fixationSets, 1)
	curSetName = p.fixationSets{curSet};
	load(sprintf('%s/fixations/%sFix.mat', p.datasetDir, curSetName));
	load(sprintf('%s/fixations/%sSize.mat', p.datasetDir, curSetName));		% Benchmark faster by skipping reading image size info from jpg files.
	outFileName = sprintf('%s/fixations/%sFixmap.mat', p.datasetDir, curSetName);
    imgNum = size(fixCell, 1);
    for curImg = 1:imgNum
        
        curFixmap = zeros(sizeData(curImg,:));
        posFix = int16(fixCell{curImg});
        for curFix = 1:size(posFix,1)
%             fprintf('%s - %d - %d\n',curSetName, curImg,curFix);
            if posFix(curFix,1) > 0 && posFix(curFix,2) >0
                curFixmap(posFix(curFix,1),posFix(curFix,2)) = 1;
            end
        end
         fixmapCell{curImg} = curFixmap;
    end
    save(outFileName,'fixmapCell','-v7.3');
    clear fixmapCell;
end