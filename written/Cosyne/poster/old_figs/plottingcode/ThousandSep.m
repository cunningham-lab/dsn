function out = ThousandSep(in)
import java.text.*
v = DecimalFormat;
for i = 1:numel(in)
    out{i} = char(v.format(in(i)));
end