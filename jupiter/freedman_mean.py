import math
import torch

class FreedmanMean(torch.nn.Module):
    def __init__(self, scale: float=1.0, metallicity: float=0.0):
        super().__init__()
        self.scale = scale
        self.metallicity = metallicity

    def forward(self, 
                conc: torch.Tensor, 
                temp: torch.Tensor, 
                pres: torch.Tensor, 
                dens: torch.Tensor) -> torch.Tensor:
        # coefficient from Richard S. Freedman 2014. APJS
        c1 = 10.602
        c2 = 2.882
        c3 = 6.09e-15
        c4 = 2.954
        c5 = -2.526
        c6 = 0.843
        c7 = -5.490
        c13 = 0.8321

        # Piecewise coefficients
        c8  = torch.where(temp < 800., torch.full_like(temp, -14.051), torch.full_like(temp, 82.241))
        c9  = torch.where(temp < 800., torch.full_like(temp,  3.055),  torch.full_like(temp, -55.456))
        c10 = torch.where(temp < 800., torch.full_like(temp,  0.024),  torch.full_like(temp, 8.754))
        c11 = torch.where(temp < 800., torch.full_like(temp,  1.877),  torch.full_like(temp, 0.7048))
        c12 = torch.where(temp < 800., torch.full_like(temp, -0.445),  torch.full_like(temp, -0.0414))

        # log10 of pressure (Pa → dyn/cm^2) and temperature
        logp = torch.log10(pres * 10.0)
        logT = torch.log10(temp)

        # Clamp in-place
        logp.clamp_(min=0.0)               # 1 microbar to 300 bar
        logT.clamp_(min=math.log10(75.0))  # 75 K to 4000 K

        # Eqn 4 (klowp)
        klowp = (
            c1 * torch.atan(logT - c2)
            - c3 / (logp + c4) * torch.exp(torch.pow(logT - c5, 2.0))
            + c6 * self.metallicity
            + c7
        )

        # Eqn 5 (khigp)
        khigp = (
            c8
            + c9 * logT
            + c10 * logT.pow(2.0)
            + logp * (c11 + c12 * logT)
            + c13 * self.metallicity
            * (0.5 + 1.0 / math.pi * torch.atan((logT - 2.5) / 0.2))
        )

        # Combine results
        result = torch.pow(10.0, klowp) + torch.pow(10.0, khigp)  # cm^2/g

        # Final scaled opacity (1/m)
        return self.scale * 0.1 * (dens * result).unsqueeze(0).unsqueeze(-1)

if __name__ == "__main__":
    model = FreedmanMean()
    scripted = torch.jit.script(model)
    scripted.save("freedman_mean.pt")
