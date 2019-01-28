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

Some neural circuit models from theoretical neuroscience are already implemented as built-in system classes in the DSN library.

"""

footer_text = """

# References #

Dipoppa, Mario, et al. *[Vision and locomotion shape the interactions between neuron types in mouse visual cortex](https://www.sciencedirect.com/science/article/pii/S0896627318302435){:target="_blank"}*. Neuron 98.3 (2018): 602-615. <a name="Dipoppa2018Vision"></a>

Pfeffer, Carsten K., et al. [Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons](https://www.nature.com/articles/nn.3446){:target="_blank"}*." Nature neuroscience 16.8 (2013): 1068. <a name="Pfeffer2013Inhibition"></a>

Mastrogiuseppe, Francesca, and Srdjan Ostojic. *[Linking connectivity, dynamics, and computations in low-rank recurrent neural networks](https://www.sciencedirect.com/science/article/pii/S0896627318305439){:target="_blank"}*. Neuron 99.3 (2018): 609-623. <a name="Mastrogiuseppe2018Linking"></a>


"""



destfile = open(destfn, "w");
destfile.write(header_text);

sourcefile = open(sourcefn, "r");
lines = [];
i = 1;
for line in sourcefile:
	if (line[:3] == "## "):
		# write a big line to separate classes
		### destfile.write('\n*****\n');
		# fine underscores in class names
		proc_line = line.replace('_', '\_');
		# add ref tag
		line = line[:2] + ' <a name="' + line[3:-1] + '"> </a> ' + proc_line[3:]
	elif (line[:4] == "### "):
		line = line.replace('_', '\_');
	i += 1;
	destfile.write(line);


destfile.write(footer_text);

