import dsn.util.systems as dsnsys
import re, inspect

# system_strs = ['linear_2D', 'R1RNN_input', 'V1_circuit'];
system_strs = ["linear_2D", "V1_circuit"]
num_system_strs = len(system_strs)


def doc2md(docstring, keywords):
    docstrings = docstring.split("\n")
    num_strings = len(docstrings)

    i = 0
    key_ind = 0
    num_keys = len(keywords)
    key_len = len(keywords[0])
    found_key = False
    more_keys = True

    markdown = []
    while i < num_strings:
        docstring_i = re.sub("\s+", " ", docstrings[i])
        if i > 0:
            docstring_i = docstring_i[1:]

        if more_keys and (docstring_i[:key_len] == keywords[key_ind]):
            docstring_i = (
                "**" + docstring_i[:key_len] + "**" + docstring_i[key_len:] + "\\\\"
            )
            key_ind += 1
            more_keys = not (key_ind == num_keys)
            if not found_key:
                found_key = True
            if more_keys:
                key_len = len(keywords[key_ind])

        if found_key:
            if "):" in docstring_i:
                words = docstring_i.split(" ")
                line = "**" + words[0] + "** " + words[1] + " *"
                for j in range(2, len(words) - 1):
                    line += words[j] + " "

                line += words[-1] + "*"
                docstring_i = line

        markdown.append(docstring_i)
        i += 1

    return markdown


def parse_members(members):
    markdown = []
    keywords = ["Args", "Returns"]
    for i in range(len(members)):
        name, method = members[i]
        if not (name[0:2] == "__"):
            markdown.append("### " + name + " ###")
            markdown += doc2md(method.__doc__, keywords)
            markdown.append("\n")
    return markdown


print("# system #")
docstring = dsnsys.system.__doc__
keywords = ["Attributes"]
markdown = doc2md(docstring, keywords)
for i in range(len(markdown)):
    print(markdown[i])

print("*****\n")

for i in range(num_system_strs):
    print("# " + system_strs[i] + " #")
    system_class = dsnsys.system_from_str(system_strs[i])
    docstring = system_class.__doc__
    keywords = ["Attributes"]
    markdown = doc2md(docstring, keywords)
    markdown += parse_members(inspect.getmembers(system_class))
    for i in range(len(markdown)):
        print(markdown[i])

print("*****\n")
