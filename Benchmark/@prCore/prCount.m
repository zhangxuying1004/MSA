function [prec, recall] = prCount(gtMask, curSMap, doSmoothing, p)
%PRCOUNT Summary of this function goes here
%   Detailed explanation goes here

[gtH, gtW] = size(gtMask);
[algH, algW] = size(curSMap);
if gtH~=algH || gtW~=algW
	curSMap = imresize(curSMap, [gtH, gtW]);
end

if doSmoothing
	kSize = norm([gtH, gtW]).*p.defaultSigma;
	curH = fspecial('gaussian', round([kSize, kSize]*5), kSize);
	curSMap = mat2gray(imfilter(curSMap, curH));
	cbImg = mat2gray(fspecial('gaussian', [gtH, gtW], gtW*0.4));
	curSMap = (curSMap + cbImg)./2;
end


gtMask = gtMask>=p.gtThreshold;
gtInd = find(gtMask(:)>0);
gtCnt = sum(sum(gtMask));


if gtCnt==0
	prec = [];
	recall = [];
	return;
end


hitCnt = zeros(p.thNum, 1);
algCnt = zeros(p.thNum, 1);


for curTh = 1:p.thNum
	thSMap = curSMap>=p.thList(curTh);
	hitCnt(curTh) = sum(thSMap(gtInd));
	algCnt(curTh) = sum(sum(thSMap));
end

prec = hitCnt./(algCnt+eps);
recall = hitCnt./gtCnt;

end

