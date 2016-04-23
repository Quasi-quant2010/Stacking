# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'

import os
from xml.dom import minidom, Node
import platform
import sys


class ConfigException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class PlatformInfo:
    def __init__(self, name=None, family=None, arch=None, version=None):
        self.name = name
        self.family = family
        self.arch = arch
        self.version = version

    @staticmethod
    def check(str1, str2):
        ok = 1
        if str1 and str2:
            flag = str1.lower().find(str2.lower()) >= 0
            if not flag:
                flag = str2.lower().find(str1.lower()) >= 0
            ok = ok and flag
        return ok

    def match(self, name, family, arch, version):
        flag = PlatformInfo.check(self.name, name)
        flag = flag and PlatformInfo.check(self.family, family)
        flag = flag and PlatformInfo.check(self.arch, arch)
        flag = flag and PlatformInfo.check(self.version, version)
        return flag


def _get_tag_name_and_text(element):
    if element is None or element.firstChild is None:
        return None, None
    if element.firstChild.nodeType == Node.TEXT_NODE:
        return element.tagName, element.firstChild.data
    return element.tagName, None


def _get_text(element):
    return _get_tag_name_and_text(element)[1]


def _get_first_element_by_tag_name(ele, name):
    elements = ele.getElementsByTagName(name)
    if elements and len(elements) > 0:
        return elements[0]
    return None


def convert_doc_to_config(doc):
    config = dict()
    for element in doc.childNodes:
        key, val = _get_tag_name_and_text(element)
        if val:
            config[key] = val
    return config


def set_config_file_and_load(fname = 'config.xml'):
    _config_file_name = fname
    config_xml = minidom.parse(_config_file_name)
    _global_configs = convert_doc_to_config(config_xml.firstChild)
    del _global_configs['profiles']
    _profiles = dict()
    _activations = dict()
    profiles = config_xml.getElementsByTagName('profile')
    if profiles:
        for profile in profiles:
            # profile id
            id_element = _get_first_element_by_tag_name(profile, 'id')
            if id_element is None:
                raise ConfigException(
                    'Profile <id> tag not found in config file "%s"!' % _config_file_name)
            idstr = _get_text(id_element)
            _profiles[idstr] = dict()
            _profiles[idstr].update(_global_configs)

            # activation
            activation = _get_first_element_by_tag_name(profile, 'activation')
            if activation:
                os = _get_first_element_by_tag_name(activation, 'os')
                if os:
                    name = _get_first_element_by_tag_name(os, 'name')
                    name = _get_text(name)
                    family = _get_first_element_by_tag_name(os, 'family')
                    family = _get_text(family)
                    arch = _get_first_element_by_tag_name(os, 'arch')
                    arch = _get_text(arch)
                    version = _get_first_element_by_tag_name(os, 'version')
                    version = _get_text(version)
                    _activations[idstr] = PlatformInfo(name=name, family=family, arch=arch,
                                                                     version=version)
            # properties
            properties = _get_first_element_by_tag_name(profile, 'properties')
            if properties is None:
                raise ConfigException(
                    'Profile <properties> tag not found in config file "%s"!' % _config_file_name)
            _profiles[idstr].update(convert_doc_to_config(properties))
            # merge global configs into different profiles
    return _config_file_name, _global_configs, _profiles, _activations


class Config:
    _profile_name = None
    _global_configs = None
    _profiles = dict()
    _activations = None
    _activated_profile = None
    _default_config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.xml')#../config.xml
    try:
        # print _default_config_file_path
        _config_file_name, _global_configs, _profiles, _activations = set_config_file_and_load(fname = _default_config_file_path)
    except Exception, e:
        print 'Load config file "%s" error! Please check the xml file, or call ' \
              'ConfigManager.set_config_file_and_load(<your_config_file>) explicitly.'\
              % _default_config_file_path
        print sys.exc_info()
        exit(1)
        pass

    @staticmethod
    def set_config_profile(profile_name):
        if profile_name in Config._profiles:
            Config._profile_name = profile_name
            Config._activated_profile = Config._profiles[profile_name]
        else:
            raise ConfigException('Profile %s not found! Please check your config file! "%s"'
                                  % (profile_name, Config._config_file_name))

    @staticmethod
    def get_config_dict():
        if Config._activated_profile:
            return Config._activated_profile
        if Config._profile_name is None:
            name = platform.platform()
            family = platform.system()
            arch = platform.machine()
            version = platform.version()
            for profile, platform_info in Config._activations.iteritems():
                if platform_info.match(name, family, arch, version):
                    # use the satisfied config
                    Config._activated_profile = Config._profiles[profile]
                    return Config._activated_profile
            if len(Config._profiles) > 0:
                # use any available config
                Config._activated_profile = Config._profiles.values()[0]
                return Config._activated_profile
            else:
                # use global config
                Config._activated_profile = Config._global_configs
                return Config._activated_profile

    @staticmethod
    def get_string(key):
        config_dict = Config.get_config_dict()
        if key not in config_dict:
            raise ConfigException('Attribute "%s" of config not found! Please check your config file! ' % key)
        return config_dict[key]

    @staticmethod
    def get_int(key):
        return int(Config.get_string(key))

    @staticmethod
    def get_float(key):
        return float(Config.get_string(key))

    @staticmethod
    def set_config_file_and_load(fname=None):
        if fname is None:
            fname = Config._default_config_file_path
        Config._config_file_name, \
        Config._global_configs, \
        Config._profiles, \
        ConfigManager_activations = set_config_file_and_load(fname = fname)

    @staticmethod
    def print_config_dict(config_dic=None):
        if config_dic is None:
            config_dic = Config._activated_profile
        for k, v in config_dic.iteritems():
            print '%s:\t%s' % (k, v)

if __name__=='__main__':
    Config.set_config_profile('oyz_linux')
    Config.print_config_dict()
    pass

