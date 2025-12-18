class AttributeSetter:
    @staticmethod
    def set_attributes(obj, attributes: dict):
        for key, value in attributes.items():
            setattr(obj, key, value)