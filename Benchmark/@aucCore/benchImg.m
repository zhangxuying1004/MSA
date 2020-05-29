function tmpAUC = benchImg(rawSMap, posFix, negFix, kSizeList)
%BENCHIMG computes single image AUC based on pos&neg fixes under 
%different smoothing kernels. CPU version.
%   Xiaodi Hou <xiaodi.hou@gmail.com>, 2014
%   Please email me if you find bugs or have questions.

kNum = length(kSizeList);
tmpAUC = zeros(kNum, 1);
for curK = 1:kNum
	kSize = kSizeList(curK);
	if kSize==0
		smoothSMap = rawSMap;
	else
		curH = fspecial('gaussian', round([kSize, kSize]*5), kSize);	% construct blur kernel
		smoothSMap = imfilter(rawSMap, curH);
	end
	% compute AUC
	tmpAUC(curK) = aucCore.compAUC(smoothSMap, posFix, negFix);		% highly optimized AUC computation
end

end

