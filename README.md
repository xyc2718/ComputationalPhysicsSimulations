# project for the Computational Physics Simulation

* These are projects for the Computational Physics Simulation Laboratory course at Fudan University for the Fall 2024 semester.

### Elastic Constant for Al,Cu and Diamond in 0K and finite temperature
We calculate the elastic properties of Al,Cu and diamond in 0K using structure optimization and in finite temperature using MD simulation.

EMA potential of Al and Cu are implemented in ema.jl 

SW potential for diamond are implemented in sw.jl

Different MD simulation method for NPT and NVT are implemented in MD.jl

* include:
* MD for NPT ensemble:

    Andersen Nose-Hoover method: Intergrate by 3 order Runge Kuta or multistep of Liouville operator and Tort Decomposition
  
    Multistep of Andersen Langvin method

* NVT ensemble:
    Nose-Hoover method:Intergrate by 3 order Runge Kuta 

multi-threaded force computation is implemented on the CPU.

###  The Nuclear Quantum Effects of Water
We calculate the Zero-Point Energy,Tunneling,Vibrational Spectrum,Infrared Spectrum on Water molecule

This part extends the program from the first task，include:

* Supported molecular structures and the interaction of multiple force fields and external potential fields.
* TIP3P Potential For Water Model
* PIMD with Thermostat PILE-L
* Kinetic energy estimation in PIMD
* Statistics of RDF and Particle Occurrence Probability
  
(There are still some issues with the calculation of infrared and vibrational spectra,so for now, I am using i-pi to handle the spectral calculations.)




