function tmpAUC = benchImgGPU(rawSMap, posFix, negFix, kSizeList)
%BENCHIMGGPU computes single image AUC based on pos&neg fixes under
%different smoothing kernels. GPU version.
%   Xiaodi Hou <xiaodi.hou@gmail.com>, 2014
%   Please email me if you find bugs or have questions.

kNum = length(kSizeList);
rawSMap = gpuArray(rawSMap);
tmpAUC = zeros(kNum, 1);
for curK = 1:kNum
	kSize = kSizeList(curK);
	if kSize==0
		smoothSMap = gather(rawSMap);
	else
		curH = fspecial('gaussian', round([kSize, kSize]*5), kSize);	% construct blur kernel
		curH = gpuArray(curH);
		smoothSMap = gather(filter2(curH, rawSMap));
	end
	tmpAUC(curK) = aucCore.compAUC(smoothSMap, posFix, negFix);		% highly optimized AUC computation
end

end

