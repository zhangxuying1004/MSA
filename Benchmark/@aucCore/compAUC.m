function curAUC = compAUC( salMap, posFix, negFix )
%COMPAUC Summary of this function goes here
%   Highly optimized AUC computation.
%   Xiaodi Hou <xiaodi.hou@gmail.com>, 2014
%   Please email me if you find bugs or have questions.

%% Quantize all samples.
if ~isa(salMap, 'uint8')
	salMap = mat2gray(salMap);
	salMap = uint8(salMap.*255);
end

%% Exclude invalid pos & neg samples
[imgH, imgW] = size(salMap);
posFix = round(posFix(:, 1:2));
negFix = round(negFix(:, 1:2));
validPos = posFix(:,1)>0 & posFix(:,2)>0 & posFix(:,1)<=imgH & posFix(:,2)<=imgW;
validNeg = negFix(:,1)>0 & negFix(:,2)>0 & negFix(:,1)<=imgH & negFix(:,2)<=imgW;
posFix = posFix(validPos, :);
negFix = negFix(validNeg, :);
posInd = sub2ind([imgH, imgW], posFix(:,1), posFix(:,2));
negInd = sub2ind([imgH, imgW], negFix(:,1), negFix(:,2));
posData = salMap(posInd);
negData = salMap(negInd);

posPtNum = size(posData, 1);
negPtNum = size(negData, 1);

%% Determine threshold list by taking all unique values of pos & neg samples.
thList = accumarray([posData; negData]+1, 1, [255, 1]);		% MATLAB's unique() is absurdily slow for uint8!
thList = [0; find(thList)];
thNum = length(thList);


%% Threshold and count
posData = sort(posData, 'descend');		% sort & find is faster than thresholding many times
negData = sort(negData, 'descend');

rocCurve = zeros(thNum, 2);
for curThNum = 1:thNum
	curT = thList(curThNum);
	tpCount = find(posData>=curT, 1, 'last');
	fpCount = find(negData>=curT, 1, 'last');
	if isempty(tpCount)
		tpCount = 0;
	end
	if isempty(fpCount)
		fpCount = 0;
	end
	rocCurve(curThNum,:) = [tpCount, fpCount];
end
rocCurve = bsxfun(@rdivide, rocCurve, [posPtNum, negPtNum]);		% normalize by total counts.
curAUC = aucCore.areaROC(rocCurve);

end

