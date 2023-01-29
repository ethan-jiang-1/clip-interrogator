from clip_interrogator.clip_interrogator_hacked import Config, InterrogatorPet
from datetime import datetime
from PIL import Image
import logging
from pprint import pprint


class XciInterrogator:
    @classmethod
    def create_ci(cls, ci_mode="blip:clip", logger: logging.Logger = None) -> InterrogatorPet:
        d0 = datetime.now()
        config = Config()
        config.ci_mode = ci_mode
        
        ci = InterrogatorPet(config)
        # TODO: (ethan) no cache the txt to pkl -- as our txt is still changing
        # should turn off once we have label txt built
        ci.config.cache_path = None 

        dt = datetime.now() - d0
        msg = f"create_ci ci_mode: '{ci_mode}' takes {round(dt.total_seconds(), 2)} sec"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return ci


class XciPromptGenerator:
    @classmethod
    def inference(cls, ci:InterrogatorPet, image: Image, mode:str, logger: logging.Logger = None, inspect=False):
        d0 = datetime.now()
        image = image.convert('RGB')
        if mode == 'caption':
            prompt = ci.interrogate_caption(image)
        else:
            prompt, label_dict = ci.interrogate_full(image)
            if inspect:
                pprint(label_dict)
        dt = datetime.now() - d0
        msg = f"inference takes {round(dt.total_seconds(), 2)} sec"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

        return prompt

class XciPetPromptGenerator:
    @classmethod
    def _replace_pet_with_fireword_in_prmpt(cls, prompt):
        # TODO: (ethan) would be nice if we use regular expressions
        # so that we would not replace wrong string such as cattle, dogging 
        prompt = prompt.replace('cat', 'qzectbumo')
        prompt = prompt.replace('kitten', 'qzectbumo')
        prompt = prompt.replace('dog', 'qzectbumo')
        prompt = prompt.replace('puppy', 'qzectbumo')
        return prompt

    @classmethod
    def inference_with_txt(cls, ci:InterrogatorPet, image_path:str, logger: logging.Logger = None):
        image = Image.open(image_path)   
        prompt = cls.inference(ci, image, 'full', logger=logger)
        txt_path = image_path.split(".")[0] + '.txt'
        with open(txt_path, 'w') as f:
            prompt = cls._replace_pet_with_fireword_in_prmpt(prompt)
            if logger is not None:
                logger.info(prompt)
            else:
                print(prompt)
            f.write(prompt)
        return txt_path
