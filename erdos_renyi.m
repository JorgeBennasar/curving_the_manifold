function G = erdos_renyi(n,p)
 
G = rand(n,n) < p;
G = triu(G,1);
G = G + G';

end