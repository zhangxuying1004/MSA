function A = areaROC( p )

bs = unique(p(:,2));
Nbs = length(bs);
pnew = zeros(Nbs,2);
for i=1:Nbs,
    pnew(i,:) = [max(p(p(:,2)==bs(i),1)) bs(i)];
end

p = pnew;

xy = sortrows([p(:,2) p(:,1)]);

x = xy(:,1);
y = xy(:,2);

x = [0; x; 1];
y = [0; y; 1];

A = trapz( x , y );

