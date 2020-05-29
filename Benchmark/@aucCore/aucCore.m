classdef aucCore
	properties
	end
	
	methods (Static)
		areaSize = areaROC(rocCurve);
		aucScore = compAUC(salMap, posFix, negFix);
		allNegFix = genNegFix(fixCell, sizeData);
		tmpAUC = benchImg(rawSalMap, posFix, negFix, kSizeList);
		tmpAUC = benchImgGPU(rawSalMap, posFix, negFix, kSizeList);
	end
	
end

