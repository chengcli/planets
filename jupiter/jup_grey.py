import torch

class JupGasVisible(torch.nn.Module):
    """
    GreyOpacity class for handling grey opacity calculations.

    The class signature and variable dimensions are strict.
    The class must have a forward method that takes a concentration tensor
    and returns an opacity tensor.

    The concentration vector is 3D with dimensions (ncol, nlyr, nspecies),
    where ncol is the number of columns, nlyr is the number of layers,
    and nspecies is the number of species.

    The opacity tensor is 4D with dimensions (nwave, ncol, nlyr, nprop),
    where nwave is the number of wavelengths, ncol is the number of columns,
    nlyr is the number of layers, and nprop is the number of optical properties.

    The first optical property is the total extinction cross-section [m^2/mol].
    The second optical property is the single scattering albedo.
    Starting from the third, the optical properties are phase function moments
    (excluding the zero-th moment).

    This class is later compiled to a TorchScript file using the `pyharp.compile` function.
    """
    def __init__(self, scale: float=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, 
                conc: torch.Tensor,
                temp: torch.Tensor,
                pres: torch.Tensor, 
                dens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conc (torch.Tensor): shape (ncol, nlyr, nspecies) [mol/m^3]
            temp (torch.Tensor): shape (ncol, nlyr) [K]
            pres (torch.Tensor): shape (ncol, nlyr) [Pa]
            dens (torch.Tensor): shape (ncol, nlyr) [kg/m^3]
            mu (torch.Tensor): shape (ncol, nlyr) [molecular weight, kg/mol]

        Returns:
            Optical properties (torch.Tensor): shape (1, ncol, nlyr, 1)
        """
        ncol, nlyr = conc.shape[0], conc.shape[1]

        # visible opacity with Jupiter haze (ncol, nlyr)
        strong_ch4 = 5.e-3 * pres.pow(-0.5)

        # visible opacity with Jupiter haze
        weak_ch4 = 1.e-3;

        return self.scale * dens * (strong_ch4 + weak_ch4).unsqueeze(0).unsqueeze(-1)

class JupGasIR(torch.nn.Module):
    def __init__(self, scale: float=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, 
                conc: torch.Tensor, 
                temp: torch.Tensor, 
                pres: torch.Tensor, 
                dens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conc (torch.Tensor): shape (ncol, nlyr, nspecies) [mol/m^3]
            temp (torch.Tensor): shape (ncol, nlyr) [K]
            pres (torch.Tensor): shape (ncol, nlyr) [Pa]
            dens (torch.Tensor): shape (ncol, nlyr) [kg/m^3]
            mu (torch.Tensor): shape (ncol, nlyr) [molecular weight, kg/mol]

        Returns:
            Optical properties (torch.Tensor): shape (1, ncol, nlyr, 1)
        """
        ncol, nlyr = conc.shape[0], conc.shape[1]

        # IR opacity from hydrocarbons and haze
        jstrat = 8.e-4 * pres.pow(-0.5);

        # infrared opacity with Jupiter haze
        cia = 2.e-8 * pres;

        return self.scale * dens * (cia + jstrat).unsqueeze(0).unsqueeze(-1)


if __name__ == "__main__":
    model = JupGasVisible()
    scripted = torch.jit.script(model)
    scripted.save(f"jup-gas-visible.pt")

    model = JupGasIR()
    scripted = torch.jit.script(model)
    scripted.save(f"jup-gas-ir.pt")
