function allNegFix = genNegFix( fixCell, sizeData )

imgNum = size(sizeData, 1);

allNegFix = cell(imgNum, 1);
for curImgNum = 1:imgNum
	curFix = fixCell{curImgNum}(:, 1:2);
	curCenter = round(sizeData(curImgNum, :)/2);
	allNegFix{curImgNum} = bsxfun(@minus, curFix, curCenter);
end


end

