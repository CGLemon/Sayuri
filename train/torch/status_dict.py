import packaging
import packaging.version
import torch
import os

class StatusDict(dict):

    MODEL_KEY = "model"
    SWA_KEY = "swa_model"
    SWA_COUNT_KEY = "swa_count"
    OPTIM_KEY = "optimizer"
    STEPS_KEY = "steps"
    SAMPLES_KEY = "samples"
    JSON_KEY = "json_str"

    MODULE_KEY_SET = [MODEL_KEY, SWA_KEY, OPTIM_KEY]
    NUMBER_KEY_SET = [SWA_COUNT_KEY, STEPS_KEY, SAMPLES_KEY]

    def __init__(self):
        super(StatusDict, self).__init__()

    def load(self, filename, device=torch.device("cpu")):
        if not os.path.isfile(filename):
            raise Exception("can't load the checkpoint from {}".format(filename))

        self.clear()
        if packaging.version.parse(torch.__version__) > packaging.version.parse("2.3.0"):
            self.update(torch.load(filename, map_location=device, weights_only=False))
        else:
            self.update(torch.load(filename, map_location=device))

    def save(self, filename):
        torch.save(self, filename)

    def get_(self, key, default=None, process_fn=None):
        result = self.get(key, default)
        if process_fn:
            result = process_fn(result)
        return result

    def fancy_get(self, key):
        if key in self.MODULE_KEY_SET:
            return self.get_(key, None)
        if key in self.NUMBER_KEY_SET:
            return self.get_(key, default=0)
        if key == self.JSON_KEY:
            json_str = self.get_(
                key,
                None,
                process_fn = lambda x: x if x else exec('raise Exception("json is NONE")')
            )
            return json_str
        raise Exception("invalid key")

    def set_(self, key, value, process_fn=None):
        if process_fn:
            value = process_fn(value)
        self[key] = value

    def fancy_set(self, key, value):
        if key in self.MODULE_KEY_SET:
            self.set_(key, value, lambda x: x.state_dict())
        elif key in self.NUMBER_KEY_SET:
            self.set_(key, value, lambda x: x if x else 0)
        elif key == self.JSON_KEY:
            self.set_(key, value)
        else:
            raise Exception("invalid key")

    def set_module(self, key, module):
        self.fancy_set(key, module)

    def load_module(self, key, module):
        state_dict = self.fancy_get(key)
        if state_dict:
            module.load_state_dict(state_dict)
