import os, sys
import pickle

# define the function to create the S parameter table
def export_S_params_report(hfss, filename, setup_name, sweep_name, solution_directory=""):

	# if not hfss.setups[0].sweeps[0].is_solved:
	# 	print("sweep is not solved")
	# 	return

	filepath= os.path.join(solution_directory, filename)

	oModule = hfss.odesign.GetModule("ReportSetup")
	report_name = "my S params"
	oModule.CreateReport(report_name, "Terminal Solution Data", "Data Table", setup_name+" : "+sweep_name, 
		[
			"Domain:="		, "Sweep"
		], 
		[
			"Freq:="		, ["All"],
			"probeY:="		, ["Nominal"],
			"probeX:="		, ["Nominal"],
			"$trace_length:="	, ["Nominal"],
			"$substrate_thickness:=", ["Nominal"],
			"$substrate_width:="	, ["Nominal"],
			"$trace_width:="	, ["Nominal"],
			"$trace_thickness:="	, ["Nominal"],
			"$GND_thickness:="	, ["Nominal"],
			"$probe_radius:="	, ["Nominal"],
			"$probe_height:="	, ["Nominal"],
			"$probe_positionX:="	, ["Nominal"],
			"$probe_positionY:="	, ["Nominal"],
			"$r:="			, ["Nominal"]
		],  
		[
			"X Component:="		, "Freq",
			"Y Component:="		, ["re(St(trace_T1,trace_T1))","im(St(trace_T1,trace_T1))","re(St(trace_T2,trace_T1))","im(St(trace_T2,trace_T1))",
									"re(St(trace_T1,probe))","im(St(trace_T1,probe))"]
		])

	# export
	oModule.ExportToFile(report_name, filepath, True)
	
	# delete report
	oModule.DeleteReports([report_name])


def export_prop_characteristics(hfss, active_design, filename="Propagation_Characteristics_Q2D", solution_dir = ""):
	oProj = hfss.oproject
	oDesign = oProj.SetActiveDesign(active_design)
	oModule = oDesign.GetModule("ReportSetup")

	report_name = "Propagation Characteristics"

	file_path = os.path.join(solution_dir, filename+".csv")

	setup_name = "Single"
	sweep_name = "InterpolatingHF" 


	oModule.CreateReport(report_name, "Matrix", "Data Table", "{} : {}".format(setup_name, sweep_name), 
		[
			"Context:="		, "Original"
		], 
		[
			"Freq:="		, ["All"],
			"$trace_length:="	, ["Nominal"],
			"$substrate_thickness:=", ["Nominal"],
			"$substrate_width:="	, ["Nominal"],
			"$trace_width:="	, ["Nominal"],
			"$trace_thickness:="	, ["Nominal"],
			"$GND_thickness:="	, ["Nominal"]
		], 
		[
			"X Component:="		, "Freq",
			"Y Component:="		, [
				"re(Z0(trace,trace))",
				"mag(Z0(trace,trace))",
				"im(Z0(trace,trace))", 
				"mag(AttenModal(Mode1))", 
				"mag(EpsEffModal(Mode1))", 
				"mag(LambdaEffModal(Mode1))", 
				"2*pi/mag(LambdaEffModal(Mode1))",
				"C(trace,trace)",
				
			]
		])
	oModule.ExportToFile(report_name, file_path, True) 
	oModule.DeleteReports([report_name])

				