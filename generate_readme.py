import os


filenames = ['welcome.rst', 'gettingstarted/installation.rst', 'gettingstarted/kerastosherpa.rst']
with open('README.rst', 'w') as outfile:
    for fname in filenames:
        with open(os.path.join('docs', fname)) as infile:
            outfile.write(infile.read())
