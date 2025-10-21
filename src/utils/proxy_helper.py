import os

import yaml


def __is_proxy_enabled() -> bool:
    config_path = os.path.join(os.path.dirname(__file__), '../../config/benchmark_config.yaml')
    absolute_path = os.path.abspath(config_path)
    print("Checking if proxy is enabled in config:", absolute_path)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Debug: print what we actually read
        proxy_config = config.get('proxy', {})
        enabled_value = proxy_config.get('enabled', False)
        return bool(enabled_value)
    except Exception as e:
        print(f"Error reading proxy config: {e}")
        return False


def set_proxy_authentication():
    if __is_proxy_enabled():
        print("Proxy is enabled: setting proxy authentication for localhost.")
        os.environ['HTTP_PROXY'] = "http://userlab:05dYF296KolRDwOjxvWs@webproxy00.sigmaspa.lan:3128/"
        os.environ['HTTPS_PROXY'] = "http://userlab:05dYF296KolRDwOjxvWs@webproxy00.sigmaspa.lan:3128/"

def set_no_proxy_localhost():
    #if __is_proxy_enabled():
    # Clear proxy environment variables for localhost connection
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
    os.environ['no_proxy'] = 'localhost,127.0.0.1'
