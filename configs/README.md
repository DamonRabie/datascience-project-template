# How to

```
import configparser

config = configparser.ConfigParser()
config.read(['configs/config.cfg', 'configs/config.dev.cfg'])

test_settings = config['TEST']

SERVER = test_settings["SERVER"]
```