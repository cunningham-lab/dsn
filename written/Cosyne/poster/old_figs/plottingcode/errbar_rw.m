function [handle] = errbar_rw(handle, err, Color)
axes(handle);
ax = gca;
h = ax.Children(1);
XData = h.XData;
YData = h.YData;
jitter = (rand(1)-0.5)/10;
handle = line([XData; XData], [YData-err; YData+err],'Color',Color);
end