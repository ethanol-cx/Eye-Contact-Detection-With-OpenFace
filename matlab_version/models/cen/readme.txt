Scripts for creating CEN patch experts from already trained models.

To create one from 300W + MultiPIE - create_cen_experts_gen.m
To create one from 300W + MultiPIE + Menpo - create_cen_experts_menpo.m

To create one used in OpenFace for C++ code:
create_cen_experts_OF.m (this uses both the general and menpo experts to create a joint one, with general ones used when menpo unavailable for that view)

To dowload pretrained models, go to:
https://www.dropbox.com/sh/o8g1530jle17spa/AADRntSHl_jLInmrmSwsX-Qsa?dl=0