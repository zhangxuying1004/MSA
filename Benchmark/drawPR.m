clear; clc
p = genParams();
load customColor.mat
close all;
%% Prep
setNum = size(p.salObjSets, 1);
algNum = size(p.salObjAlgs, 1);

fScore = zeros(algNum, setNum);

for curSetNum = 1:setNum
	figure(curSetNum); hold on;
	curSetName = p.salObjSets{curSetNum};
	
	%% Draw curves
	for curAlgNum = 1:algNum
		curAlgName = p.salObjAlgs{curAlgNum};
		outFileName = sprintf('%s/pr/%s_%s.mat', p.outputDir, curSetName, curAlgName);
		if ~exist(outFileName, 'file')
			error('File not found! %s\n', outFileName);
		end
		load(outFileName, 'prec', 'recall', 'thList');
		lh = plot(recall, prec, 'LineWidth', 2);
		set(lh,'Color', customColor(curAlgNum, :));
		set(lh,'LineStyle', customStyle{curAlgNum});
	end
	
	legend(upper(p.salObjAlgs));
	grid on;
	
	
	%% Draw points
	for curAlgNum = 1:algNum
		curAlgName = p.salObjAlgs{curAlgNum};
		outFileName = sprintf('%s/pr/%s_%s.mat', p.outputDir, curSetName, curAlgName);
		load(outFileName, 'prec', 'recall', 'thList');
		
		
		[curScore, curTh] = max((1+p.beta^2).*prec.*recall./(p.beta^2.*prec+recall));

		fScore(curAlgNum, curSetNum) = curScore;
		
		lh = scatter(recall(curTh), prec(curTh));
		set(lh,'MarkerFaceColor', customColor(curAlgNum, :));
		set(lh,'MarkerEdgeColor', customColor(curAlgNum, :));
		fprintf('Set %s, alg %s, F-Score=%.4f at th=%.2f\n', curSetName, curAlgName, curScore, thList(curTh));
	end
	
	title(sprintf('PR curve on %s dataset', upper(curSetName)), 'FontSize', 12);
	xlabel('Recall', 'FontSize', 11);
	ylabel('Precision', 'FontSize', 11);
	
	hold off;
	
	%% print to image files (quite slow)
	% 	paperPosition=[0 0 7.2 6];
	% 	set(gcf,'PaperUnits','inches','PaperPosition',paperPosition)
	% 	fileName = sprintf('%s/figures/AUC%s', p.outputDir, curSetName);
	% 	print(gcf,'-dpng', sprintf('%s.png', fileName));
	% 	print(gcf,'-depsc', sprintf('%s.eps', fileName));
end