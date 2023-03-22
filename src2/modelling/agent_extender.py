import xml.etree.ElementTree as xml
import sys

N = 3


def extract_template(root: xml.Element) -> xml.Element:
    pass


def insert_template(root: xml.Element, template: xml.Element):
    pass


def main():
    etree = xml.parse(sys.argv[1])
    root: xml.Element = etree.getroot()

    template = extract_template(root)

    for n in range(N):
        insert_template(root, template)

    output = xml.ElementTree(root)
    xml.indent(output, space='\t', level=0)
    output.write(sys.argv[2])


if __name__ == "__main__":
    main()
