sourcefn = "/Users/sbittner/Documents/dsn/docs/system_docs.md"
destfn = "/Users/sbittner/Documents/dsn/docs/systems.md"

header_text = """---
title: Systems
permalink: /systems/
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

<div class="topnav">
  <a class="active" href="../">Home</a>
  <a href="#">DSN Systems Library</a>
</div>

Many commonly used models in theoretical neuroscience are already implemented as built-in system classes in the DSN library.

"""

destfile = open(destfn, "w");
destfile.write(header_text);

sourcefile = open(sourcefn, "r");
lines = [];
i = 1;
for line in sourcefile:
	if (line[:3] == "## "):
		# write a big line to separate classes
		destfile.write('\n*****\n');
		# fine underscores in class names
		proc_line = line.replace('_', '\_');
		# add ref tag
		line = line[:2] + ' <a name="' + line[3:-1] + '"> </a> ' + proc_line[3:]
	elif (line[:4] == "### "):
		line = line.replace('_', '\_');
	i += 1;
	destfile.write(line);



