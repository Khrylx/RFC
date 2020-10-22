from lxml import etree
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from copy import deepcopy
import argparse


def create_vis_model(in_file, out_file, num=2, expert_ind=1):
    xml_parser = XMLParser(remove_blank_text=True)
    tree = parse(in_file, parser=xml_parser)
    remove_elements = ['actuator', 'contact', 'equality']
    for elem in remove_elements:
        node = tree.getroot().find(elem)
        if node is not None:
            node.getparent().remove(node)
    
    option = tree.getroot().find('option')
    flag = SubElement(option, 'flag', {'contact': 'disable'})
    option.addnext(Element('size', {'njmax': '1000'}))

    default = tree.getroot().find('default')
    default_c = SubElement(default, 'default', {'class': 'expert'})
    SubElement(default_c, 'geom', {'rgba': '0.7 0.0 0.0 1'})

    worldbody = tree.getroot().find('worldbody')
    body = worldbody.find('body')
    for i in range(1, num):
        new_body = deepcopy(body)
        if i == expert_ind:
            new_body.attrib['childclass'] = 'expert'
        new_body.attrib['name'] = '%d_%s' % (i, new_body.attrib['name'])
        for node in new_body.findall(".//body"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//joint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//site"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        worldbody.append(new_body)
    tree.write(out_file, pretty_print=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='humanoid_smpl_neutral')
    parser.add_argument('--out_model', type=str, default='humanoid_smpl_neutral_vis')
    parser.add_argument('--num', type=int, default=2)

    args = parser.parse_args()

    in_file = 'khrylib/assets/mujoco_models/%s.xml' % args.in_model
    out_file = 'khrylib/assets/mujoco_models/%s.xml' % args.out_model
    parser = XMLParser(remove_blank_text=True)
    create_vis_model(in_file, out_file)