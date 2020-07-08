### Define paths for various folders
	SCRIPTS       = function(x="") { file.path(getwd()                                        , x) }
	INPUT         = function(x="") { file.path(dirname(SCRIPTS())                 , "../models/Sentence_Based/" , x) }
	TEMP          = function(x="") { file.path(dirname(SCRIPTS())                 , "Temp/"   , x) }
	OUTPUT        = function(x="") { file.path(dirname(SCRIPTS())                 , "Output/" , x) }

### Package management
manage_packages = function() {
	
	## Install checkpoint 0.4.9 to user folder if it does not exist.
	tryCatch(
		stopifnot(packageVersion("checkpoint") == '0.4.9'),
		error = function(e) {install.packages("checkpoint", repos = "https://mran.microsoft.com/snapshot/2020-05-25")}
	)
	
	options(pkgType="binary")
	## Setup checkpoint and snapshot date
	checkpoint::checkpoint(
		checkpointLocation = "~/",
		snapshotDate = "2020-05-25", 
		R.version = "3.6.2"
	)
	
	## Import packages
	library(tidyverse)   # for data processing
	library(foreach)     # for looping
	library(openxlsx)    # for reading/writing Excel files
	library(readr)
	# library(xlsx)        # for reading/writing Excel files
	library(DBI)         # for connecting to databases
	library(data.table)  # for exporting large csv files
	library(testthat)    # for testing code
	library(lubridate)   # for working with dates
	library(readxl)      # for importing Excel file
	library(Hmisc)
	library(scales)
	library(janitor)
	library(stringi)
	library(stringr)
	
	## Print out session information (e.g., R version, imported packages)
	sessionInfo()
}

### Define function to overwrite existing Excel sheet
export_excel = function(data, filepath, sheetname) {
	
	# If file exists, open it; if it does not exist, create it
	if (file.exists(filepath)) {
		wb = openxlsx::loadWorkbook(filepath)
		
		# If sheet already exists, delete it
		if (sheetname %in% openxlsx::getSheetNames(filepath)) {
			openxlsx::removeWorksheet(wb, sheet = sheetname)
		}
	} else {
		wb = openxlsx::createWorkbook()
	}
	
	# Write out the data to the worksheet
	openxlsx::addWorksheet(wb, sheetname)
	openxlsx::writeData(wb, sheet = sheetname, data, colNames = TRUE)
	
	# Save out updated Excel file
	openxlsx::saveWorkbook(wb, filepath, overwrite = TRUE)
}

### Define function to concat two strings together
"%+%" = function(x, y) {
	paste0(x,y)
}
