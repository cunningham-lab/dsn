rsize = 24;
scatter3(X(:,3), X(:,4), X(:,2), 50*ones(N,1), y, 'o', 'filled');
zlabel('', 'FontSize', fsize);
xlabel('', 'FontSize', fsize);
ylabel('');
ylim([-.2,.55])
xlim([.1,.6])
zlim([.5, 1]);
