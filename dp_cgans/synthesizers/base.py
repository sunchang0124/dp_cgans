import torch


class BaseSynthesizer:

    def save(self, path):
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        torch.save(self, path)
        self.set_device(device_backup)
    
    def xai_discriminator(self, data_samples):
        discriminator_predict_score = self._discriminator(data_samples)
        return discriminator_predict_score

    @classmethod
    def load(cls, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(path)
        model.set_device(device)
        return model
