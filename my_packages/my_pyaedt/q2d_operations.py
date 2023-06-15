def change_material(hfss, active_design, material_name, setup_name, sweep_name):
    oProj = hfss.oproject
    oDesign = oProj.SetActiveDesign(active_design)
    oEditor = oDesign.SetActiveEditor("3D Modeler")
    oEditor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DAttributeTab",
                [
                    "NAME:PropServers", 
                    "substrate"
                ],
                [
                    "NAME:ChangedProps",
                    [
                        "NAME:Material",
                        "Value:="		, "\"{}\"".format(material_name)
                    ]
                ]
            ]
        ])
    oProj.Save()
    oDesign.Analyze(f"{setup_name} : {sweep_name}")