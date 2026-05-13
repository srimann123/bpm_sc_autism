covariate_data = readRDS("/lustre/home2/BPM/autism_brain/epilepsy_nuclei_covariates.rds") # /lustre/home/BPM/autism_brain/cas_nuclei_covariates.rds
write.csv(covariate_data, "/lustre/home2/BPM/autism_brain/epilepsy_nuclei_covariates.csv", row.names = TRUE) # /lustre/home/ramachandruss/python_single_cell/test_interface/", row.names = FALSE)


#nrow(cas_nuclei_covariates)
#sum(!is.na(cas_nuclei_covariates[, "sample_id"]))
#total = 90945
#21715 (zeros)
#69230 (non-zero)
