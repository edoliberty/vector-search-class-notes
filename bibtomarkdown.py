## This is a hacky util script to create markdown from the class bib file.
## you can run it like this
## >>  python bibtomarkdown.py class_notes/vs.bib > bib.md
## For convenience, please add urls and notes to the bib file and 
## regenerate the markdown instead of changing it directly in the README


def bibtokvdict(bib):
    lines = [line.strip() for line in bib.split('\n')[1:]]
    pairs = [line.split('=',1) for line in lines]
    return dict([ (pair[0].strip(), pair[1].strip('{}, ')) for pair in pairs])

def kvdicttomarkdown(kvdict):
    if 'title' not in kvdict:
        return ''
    s = '*'
    if 'url' in kvdict:
        s += f" [{kvdict['title']}]({kvdict['url']})"
    else:
        s += f" {kvdict.get('title','')}"
    if 'author' in kvdict:
        s += f" - {kvdict['author']}"
    if 'year' in kvdict:
        s += f" - {kvdict['year']}"
    return s

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bib_file_name", help="path to a *.bib file")
    args = parser.parse_args()

    with open(args.bib_file_name) as f:
        text = f.read()

    bibs = [s.strip() for s in text.split('\n\n') if s.startswith('@')]
    kvdicts = [bibtokvdict(bib) for bib in bibs]
    kvdicts.sort(key = lambda m:m['title'].lower())
    mds = [kvdicttomarkdown(kvdict) for kvdict in kvdicts]
    for md in mds:
        print(md)