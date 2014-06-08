<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="12008004">
	<Property Name="NI.LV.ExampleFinder" Type="Str">&lt;?xml version="1.0" encoding="UTF-8"?&gt;
&lt;ExampleProgram&gt;
&lt;Title&gt;
	&lt;Text Locale="US"&gt;Analog Input.lvproj&lt;/Text&gt;
&lt;/Title&gt;
&lt;Keywords&gt;
	&lt;Item&gt;CompactRIO&lt;/Item&gt;
	&lt;Item&gt;FPGA&lt;/Item&gt;
	&lt;Item&gt;started&lt;/Item&gt;
	&lt;Item&gt;getting&lt;/Item&gt;
	&lt;Item&gt;analog&lt;/Item&gt;
	&lt;Item&gt;input&lt;/Item&gt;
&lt;/Keywords&gt;
&lt;Navigation&gt;
	&lt;Item&gt;7253&lt;/Item&gt;
	&lt;Item&gt;7695&lt;/Item&gt;
&lt;/Navigation&gt;
&lt;FileType&gt;LV Project&lt;/FileType&gt;
&lt;Metadata&gt;
&lt;Item Name="RTSupport"&gt;LV Project RT&lt;/Item&gt;
&lt;/Metadata&gt;
&lt;ProgrammingLanguages&gt;
&lt;Item&gt;LabVIEW&lt;/Item&gt;
&lt;/ProgrammingLanguages&gt;
&lt;RequiredSoftware&gt;
&lt;NiSoftware MinVersion="8.0"&gt;LabVIEW&lt;/NiSoftware&gt; 
&lt;/RequiredSoftware&gt;
&lt;RequiredFPGAHardware&gt;
&lt;Device&gt;
&lt;Model&gt;7055&lt;/Model&gt;
&lt;Model&gt;7056&lt;/Model&gt;
&lt;Model&gt;702C&lt;/Model&gt;
&lt;Model&gt;702D&lt;/Model&gt;
&lt;Model&gt;7074&lt;/Model&gt;
&lt;Model&gt;7083&lt;/Model&gt;
&lt;/Device&gt;
&lt;/RequiredFPGAHardware&gt;
&lt;/ExampleProgram&gt;</Property>
	<Property Name="NI.Project.Description" Type="Str">This project shows how to read analog input channels from a R Series board.  It was created using a PCI-7831R.  You can use the same code for other R Series boards by creating your desired target and then cutting and pasting all items in the 'Items to Move' folder to your new target.  

This example needs to be compiled for a specific FPGA target before use.

For information on moving this example to another FPGA target, refer to ni.com/info and enter info code fpgaex.</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="CCSymbols" Type="Str">OS,Win;CPU,x86;</Property>
		<Property Name="NI.SortType" Type="Int">3</Property>
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">3363</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="MultiChannelFeedbackCooling(Host).vi" Type="VI" URL="../MultiChannelFeedbackCooling(Host).vi"/>
		<Item Name="FPGA Target 2" Type="FPGA Target">
			<Property Name="AutoRun" Type="Bool">false</Property>
			<Property Name="configString.guid" Type="Str">{24248542-9D37-4038-8848-1198046E7192}Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16{2BF25492-8F4C-44D1-90A1-848A23D223AA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16{2F798544-5F1B-40C3-BA3E-77D97EA7CFBA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16{64ED8ECE-32CA-4C9E-BDED-0338ACCD5689}Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16{757C2DE6-F834-4D06-BC0B-5366788A5350}Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16{78BAFF8E-60D1-4892-9D9A-C8077839F3CA}Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16{AEE1C408-6AAB-4CE3-8D0B-33DAA5CB703A}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E{C2CC77DD-1D0C-4F4C-A647-B52B0FBE1030}Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
			<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427EAI0Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16Connector0/AI1Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16Connector0/AI2Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16Connector0/AO0Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16Connector0/AO1Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16Connector0/AO2Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16Connector0/AO3Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
			<Property Name="Mode" Type="Int">0</Property>
			<Property Name="NI.LV.FPGA.CompileConfigString" Type="Str">PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
			<Property Name="NI.LV.FPGA.Version" Type="Int">5</Property>
			<Property Name="Resource Name" Type="Str">RIO0</Property>
			<Property Name="SWEmulationSubMode" Type="UInt">1</Property>
			<Property Name="SWEmulationVIPath" Type="Path">/E/Documents and Settings/Mike/My Documents/opt_lev/vis/singin.vi</Property>
			<Property Name="Target Class" Type="Str">PCIe-7841R</Property>
			<Property Name="Top-Level Timing Source" Type="Str">40 MHz Onboard Clock</Property>
			<Property Name="Top-Level Timing Source Is Default" Type="Bool">true</Property>
			<Item Name="Items to Move" Type="Folder">
				<Item Name="AI0" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="FPGA Target">
   <Attribute name="Arbitration">
   <Value>AlwaysArbitrate</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI0</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{2F798544-5F1B-40C3-BA3E-77D97EA7CFBA}</Property>
				</Item>
				<Item Name="Connector0/AI2" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="Arbitration">
   <Value>AlwaysArbitrate</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI2</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{2BF25492-8F4C-44D1-90A1-848A23D223AA}</Property>
				</Item>
				<Item Name="Connector0/AI1" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="FPGA Target">
   <Attribute name="Arbitration">
   <Value>AlwaysArbitrate</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AI1</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{757C2DE6-F834-4D06-BC0B-5366788A5350}</Property>
				</Item>
				<Item Name="MultiChannelfFeedbackCooling (FPGA).vi" Type="VI" URL="../MultiChannelfFeedbackCooling (FPGA).vi">
					<Property Name="BuildSpec" Type="Str">{E3FA465C-1D1D-455E-A9B5-DAEF60955343}</Property>
					<Property Name="configString.guid" Type="Str">{24248542-9D37-4038-8848-1198046E7192}Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16{2BF25492-8F4C-44D1-90A1-848A23D223AA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16{2F798544-5F1B-40C3-BA3E-77D97EA7CFBA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16{64ED8ECE-32CA-4C9E-BDED-0338ACCD5689}Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16{757C2DE6-F834-4D06-BC0B-5366788A5350}Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16{78BAFF8E-60D1-4892-9D9A-C8077839F3CA}Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16{AEE1C408-6AAB-4CE3-8D0B-33DAA5CB703A}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E{C2CC77DD-1D0C-4F4C-A647-B52B0FBE1030}Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427EAI0Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16Connector0/AI1Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16Connector0/AI2Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16Connector0/AO0Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16Connector0/AO1Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16Connector0/AO2Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16Connector0/AO3Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
					<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">E:\Documents and Settings\Mike\My Documents\opt_lev\vis\FPGA Bitfiles\MultiChannelFeed_FPGATarget2_feedbackcooling(_T5hmXv3JUos.lvbitx</Property>
				</Item>
			</Item>
			<Item Name="Connector0" Type="Folder">
				<Item Name="Connector0/AO0" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="Arbitration">
   <Value>AlwaysArbitrate</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO0</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{C2CC77DD-1D0C-4F4C-A647-B52B0FBE1030}</Property>
				</Item>
				<Item Name="Connector0/AO1" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="Arbitration">
   <Value>AlwaysArbitrate</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO1</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{64ED8ECE-32CA-4C9E-BDED-0338ACCD5689}</Property>
				</Item>
				<Item Name="Connector0/AO2" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="Arbitration">
   <Value>AlwaysArbitrate</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO2</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{78BAFF8E-60D1-4892-9D9A-C8077839F3CA}</Property>
				</Item>
				<Item Name="Connector0/AO3" Type="Elemental IO">
					<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="Arbitration">
   <Value>AlwaysArbitrate</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Connector0/AO3</Value>
   </Attribute>
</AttributeSet>
</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{24248542-9D37-4038-8848-1198046E7192}</Property>
				</Item>
			</Item>
			<Item Name="40 MHz Onboard Clock" Type="FPGA Base Clock">
				<Property Name="FPGA.PersistentID" Type="Str">{AEE1C408-6AAB-4CE3-8D0B-33DAA5CB703A}</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig" Type="Str">ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.Accuracy" Type="Dbl">100</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.ClockSignalName" Type="Str">Clk40</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MaxDutyCycle" Type="Dbl">50</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MaxFrequency" Type="Dbl">40000000</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MinDutyCycle" Type="Dbl">50</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.MinFrequency" Type="Dbl">40000000</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.NominalFrequency" Type="Dbl">40000000</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.PeakPeriodJitter" Type="Dbl">250</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.ResourceName" Type="Str">40 MHz Onboard Clock</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.SupportAndRequireRuntimeEnableDisable" Type="Bool">false</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.TopSignalConnect" Type="Str">Clk40</Property>
				<Property Name="NI.LV.FPGA.BaseTSConfig.VariableFrequency" Type="Bool">false</Property>
				<Property Name="NI.LV.FPGA.Valid" Type="Bool">true</Property>
				<Property Name="NI.LV.FPGA.Version" Type="Int">5</Property>
			</Item>
			<Item Name="rangelimiter.vi" Type="VI" URL="../rangelimiter.vi">
				<Property Name="configString.guid" Type="Str">{24248542-9D37-4038-8848-1198046E7192}Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16{2BF25492-8F4C-44D1-90A1-848A23D223AA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16{2F798544-5F1B-40C3-BA3E-77D97EA7CFBA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16{64ED8ECE-32CA-4C9E-BDED-0338ACCD5689}Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16{757C2DE6-F834-4D06-BC0B-5366788A5350}Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16{78BAFF8E-60D1-4892-9D9A-C8077839F3CA}Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16{AEE1C408-6AAB-4CE3-8D0B-33DAA5CB703A}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E{C2CC77DD-1D0C-4F4C-A647-B52B0FBE1030}Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
				<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427EAI0Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16Connector0/AI1Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16Connector0/AI2Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16Connector0/AO0Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16Connector0/AO1Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16Connector0/AO2Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16Connector0/AO3Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
			</Item>
			<Item Name="singin.vi" Type="VI" URL="../singin.vi">
				<Property Name="configString.guid" Type="Str">{24248542-9D37-4038-8848-1198046E7192}Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16{2BF25492-8F4C-44D1-90A1-848A23D223AA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16{2F798544-5F1B-40C3-BA3E-77D97EA7CFBA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16{64ED8ECE-32CA-4C9E-BDED-0338ACCD5689}Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16{757C2DE6-F834-4D06-BC0B-5366788A5350}Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16{78BAFF8E-60D1-4892-9D9A-C8077839F3CA}Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16{AEE1C408-6AAB-4CE3-8D0B-33DAA5CB703A}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E{C2CC77DD-1D0C-4F4C-A647-B52B0FBE1030}Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
				<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427EAI0Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16Connector0/AI1Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16Connector0/AI2Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16Connector0/AO0Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16Connector0/AO1Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16Connector0/AO2Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16Connector0/AO3Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
			</Item>
			<Item Name="BeadCoolerFixedZ(SubVI).vi" Type="VI" URL="../BeadCoolerFixedZ(SubVI).vi">
				<Property Name="configString.guid" Type="Str">{24248542-9D37-4038-8848-1198046E7192}Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16{2BF25492-8F4C-44D1-90A1-848A23D223AA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16{2F798544-5F1B-40C3-BA3E-77D97EA7CFBA}Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16{64ED8ECE-32CA-4C9E-BDED-0338ACCD5689}Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16{757C2DE6-F834-4D06-BC0B-5366788A5350}Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16{78BAFF8E-60D1-4892-9D9A-C8077839F3CA}Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16{AEE1C408-6AAB-4CE3-8D0B-33DAA5CB703A}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E{C2CC77DD-1D0C-4F4C-A647-B52B0FBE1030}Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
				<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427EAI0Arbitration=AlwaysArbitrate;resource=/Connector0/AI0;0;ReadMethodType=I16Connector0/AI1Arbitration=AlwaysArbitrate;resource=/Connector0/AI1;0;ReadMethodType=I16Connector0/AI2Arbitration=AlwaysArbitrate;resource=/Connector0/AI2;0;ReadMethodType=I16Connector0/AO0Arbitration=AlwaysArbitrate;resource=/Connector0/AO0;0;WriteMethodType=I16Connector0/AO1Arbitration=AlwaysArbitrate;resource=/Connector0/AO1;0;WriteMethodType=I16Connector0/AO2Arbitration=AlwaysArbitrate;resource=/Connector0/AO2;0;WriteMethodType=I16Connector0/AO3Arbitration=AlwaysArbitrate;resource=/Connector0/AO3;0;WriteMethodType=I16PCIe-7841R/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_FAMILYVIRTEX5TARGET_TYPEFPGA</Property>
			</Item>
			<Item Name="Dependencies" Type="Dependencies">
				<Item Name="vi.lib" Type="Folder">
					<Item Name="niFPGA I32xI32 MAC+ MSB.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Analysis/utilities/niFPGA I32xI32 MAC+ MSB.vi"/>
					<Item Name="niFPGA BW CU Order 4 (16-bit).vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Analysis/measure/butterworth/templates/niFPGA BW CU Order 4 (16-bit).vi"/>
					<Item Name="niLvFPGAUserSpecifiedEmulationVISupport.lvclass" Type="LVClass" URL="/&lt;vilib&gt;/rvi/customViForEmulation/niLvFPGAUserSpecifiedEmulationVISupport.lvclass"/>
				</Item>
				<Item Name="rangelimited.vi" Type="VI" URL="../rangelimited.vi"/>
				<Item Name="BeadCooler(SubVI).vi" Type="VI" URL="../BeadCooler(SubVI).vi"/>
				<Item Name="MatrixMult (SubVI).vi" Type="VI" URL="../MatrixMult (SubVI).vi"/>
				<Item Name="niFpgaExecutionStage.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/common/niFpgaExecutionStage.ctl"/>
				<Item Name="nirviEmuClasses.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/common/nirviEmuClasses.ctl"/>
				<Item Name="EIO_ResourceConfig.ctl" Type="VI" URL="/&lt;vilib&gt;/eio/EIO_ResourceConfig.ctl"/>
				<Item Name="niLvFpgaEmuInfo.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/sdk/emulation/public/niLvFpgaEmuInfo.ctl"/>
				<Item Name="niLvFpgaFormatErrorSource.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/niLvFpgaFormatErrorSource.vi"/>
				<Item Name="niFpgaGenCallStack.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/niFpgaGenCallStack.vi"/>
				<Item Name="nirviFillInErrorInfo.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviFillInErrorInfo.vi"/>
				<Item Name="nirviReportUnexpectedCaseInternalErrorHelper.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalErrorHelper.vi"/>
				<Item Name="nirviReportUnexpectedCaseInternalError (U32).vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError (U32).vi"/>
				<Item Name="nirviReportUnexpectedCaseInternalError (String).vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError (String).vi"/>
				<Item Name="nirviReportUnexpectedCaseInternalError (Bool).vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError (Bool).vi"/>
				<Item Name="nirviReportUnexpectedCaseInternalError.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError.vi"/>
				<Item Name="niLvFpgaErrorClusterFromErrorCode.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/niLvFpgaErrorClusterFromErrorCode.vi"/>
				<Item Name="niLvFPGAUserSpecifiedEmulationVISupport.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/customViForEmulation/niLvFPGAUserSpecifiedEmulationVISupport.ctl"/>
				<Item Name="niFpgaSctlEmulationGetInTimedLoop.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationGetInTimedLoop.vi"/>
				<Item Name="niFpgaSetErrorForExecOnDevCompSimple.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/niFpgaSetErrorForExecOnDevCompSimple.vi"/>
				<Item Name="niFpgaGetScratchAppInstance.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/common/niFpgaGetScratchAppInstance.vi"/>
				<Item Name="nirviEmuReportErrorAndStop.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/common/nirviEmuReportErrorAndStop.vi"/>
				<Item Name="niFpgaSctlEmulationClkInfo.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationClkInfo.ctl"/>
				<Item Name="niFpgaSctlEmulationSchedulerRegClks.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSchedulerRegClks.vi"/>
				<Item Name="nirviTagForDefaultClock.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/ClientSDK/Core/TimingSources/Configuration/Public/nirviTagForDefaultClock.vi"/>
				<Item Name="niFpgaSctlEmulationConstants.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationConstants.vi"/>
				<Item Name="niFpgaSctlEmulationIdMgrCmd.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationIdMgrCmd.ctl"/>
				<Item Name="niFpgaSctlEmulationIdMgr.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationIdMgr.vi"/>
				<Item Name="niFpgaSctlEmulationSchedulerHandleRollover.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSchedulerHandleRollover.vi"/>
				<Item Name="nirvimemoryEmulationManagerCacheLock_Operations.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCacheLock_Operations.ctl"/>
				<Item Name="nirvimemoryEmulationManagerCacheLock.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCacheLock.vi"/>
				<Item Name="nirvimemoryEmulationManagerCache_ReleaseExclusive.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCache_ReleaseExclusive.vi"/>
				<Item Name="nirvimemoryEmulationManagerCache_Operations.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCache_Operations.ctl"/>
				<Item Name="nirvimemoryEmulationManagerCache.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCache.vi"/>
				<Item Name="nirvimemoryEmulationManagerCache_GetValue.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCache_GetValue.vi"/>
				<Item Name="nirvimemoryEmulationManagerCache_MakeExclusive.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCache_MakeExclusive.vi"/>
				<Item Name="nirvimemoryEmulationManagerCache_SetValue.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Memory/Memory_Emulation/nirvimemoryEmulationManagerCache_SetValue.vi"/>
				<Item Name="niFpgaSctlEmulationFifoFullMgr.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationFifoFullMgr.vi"/>
				<Item Name="niFpgaSctlEmulationSharedResTypes.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSharedResTypes.ctl"/>
				<Item Name="niFpgaSctlEmulationSharedResource.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSharedResource.ctl"/>
				<Item Name="niFpgaSctlEmulationSharedResMgrCmd.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSharedResMgrCmd.ctl"/>
				<Item Name="niFpgaSctlEmulationResourceMgr.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationResourceMgr.vi"/>
				<Item Name="niFpgaSctlEmulationSchedulerUnRegClks.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSchedulerUnRegClks.vi"/>
				<Item Name="niFpgaSctlEmulationSchedulerGenSchedule.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSchedulerGenSchedule.vi"/>
				<Item Name="niFpgaSctlEmulationSchedulerState.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSchedulerState.ctl"/>
				<Item Name="niFpgaSctlEmulationSchedulerCommand.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationSchedulerCommand.ctl"/>
				<Item Name="niFpgaSctlEmulationScheduler.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationScheduler.vi"/>
				<Item Name="niFpgaSctlEmulationGlobalWrite.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationGlobalWrite.vi"/>
				<Item Name="niFpgaSctlEmulationRegisterWithScheduler.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaSctlEmulationRegisterWithScheduler.vi"/>
				<Item Name="niFpgaEmulationVisToLoad.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/Emulation/niFpgaEmulationVisToLoad.vi"/>
			</Item>
			<Item Name="Build Specifications" Type="Build">
				<Item Name="feedback_cooling (FPGA)" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
					<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
					<Property Name="BuildSpecDecription" Type="Str"></Property>
					<Property Name="BuildSpecName" Type="Str">feedback_cooling (FPGA)</Property>
					<Property Name="Comp.BitfileName" Type="Str">MultiChannelFeed_FPGATarget2_feedbackcooling(_T5hmXv3JUos.lvbitx</Property>
					<Property Name="Comp.CustomXilinxParameters" Type="Str"></Property>
					<Property Name="Comp.MaxFanout" Type="Int">-1</Property>
					<Property Name="Comp.RandomSeed" Type="Bool">false</Property>
					<Property Name="Comp.Version.Build" Type="Int">0</Property>
					<Property Name="Comp.Version.Fix" Type="Int">0</Property>
					<Property Name="Comp.Version.Major" Type="Int">1</Property>
					<Property Name="Comp.Version.Minor" Type="Int">0</Property>
					<Property Name="Comp.VersionAutoIncrement" Type="Bool">false</Property>
					<Property Name="Comp.Xilinx.DesignStrategy" Type="Str">balanced</Property>
					<Property Name="Comp.Xilinx.MapEffort" Type="Str">high(timing)</Property>
					<Property Name="Comp.Xilinx.ParEffort" Type="Str">standard</Property>
					<Property Name="Comp.Xilinx.SynthEffort" Type="Str">normal</Property>
					<Property Name="Comp.Xilinx.SynthGoal" Type="Str">speed</Property>
					<Property Name="Comp.Xilinx.UseRecommended" Type="Bool">true</Property>
					<Property Name="DefaultBuildSpec" Type="Bool">true</Property>
					<Property Name="DestinationDirectory" Type="Path">FPGA Bitfiles</Property>
					<Property Name="ProjectPath" Type="Path">/E/Documents and Settings/Mike/My Documents/opt_lev/vis/MultiChannelFeedbackCooling.lvproj</Property>
					<Property Name="RelativePath" Type="Bool">true</Property>
					<Property Name="RunWhenLoaded" Type="Bool">false</Property>
					<Property Name="SupportDownload" Type="Bool">true</Property>
					<Property Name="SupportResourceEstimation" Type="Bool">true</Property>
					<Property Name="TargetName" Type="Str">FPGA Target 2</Property>
					<Property Name="TopLevelVI" Type="Ref">/My Computer/FPGA Target 2/Items to Move/MultiChannelfFeedbackCooling (FPGA).vi</Property>
				</Item>
			</Item>
		</Item>
		<Item Name="singin.vi" Type="VI" URL="../singin.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Simple Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Simple Error Handler.vi"/>
				<Item Name="subTimeDelay.vi" Type="VI" URL="/&lt;vilib&gt;/express/express execution control/TimeDelayBlock.llb/subTimeDelay.vi"/>
				<Item Name="niLvFPGAUserSpecifiedEmulationVISupport.lvclass" Type="LVClass" URL="/&lt;vilib&gt;/rvi/customViForEmulation/niLvFPGAUserSpecifiedEmulationVISupport.lvclass"/>
				<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
			</Item>
			<Item Name="NiFpgaLv.dll" Type="Document" URL="NiFpgaLv.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="MultiChannelFeed_FPGATarget2_feedbackcooling(_T5hmXv3JUos.lvbitx" Type="Document" URL="../MultiChannelFeed_FPGATarget2_feedbackcooling(_T5hmXv3JUos.lvbitx"/>
			<Item Name="niFpgaExecutionStage.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/common/niFpgaExecutionStage.ctl"/>
			<Item Name="nirviEmuClasses.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/common/nirviEmuClasses.ctl"/>
			<Item Name="EIO_ResourceConfig.ctl" Type="VI" URL="/&lt;vilib&gt;/eio/EIO_ResourceConfig.ctl"/>
			<Item Name="niLvFpgaEmuInfo.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/eio/sdk/emulation/public/niLvFpgaEmuInfo.ctl"/>
			<Item Name="niLvFpgaFormatErrorSource.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/niLvFpgaFormatErrorSource.vi"/>
			<Item Name="niFpgaGenCallStack.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/niFpgaGenCallStack.vi"/>
			<Item Name="nirviFillInErrorInfo.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviFillInErrorInfo.vi"/>
			<Item Name="nirviReportUnexpectedCaseInternalErrorHelper.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalErrorHelper.vi"/>
			<Item Name="nirviReportUnexpectedCaseInternalError (U32).vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError (U32).vi"/>
			<Item Name="nirviReportUnexpectedCaseInternalError (String).vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError (String).vi"/>
			<Item Name="nirviReportUnexpectedCaseInternalError (Bool).vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError (Bool).vi"/>
			<Item Name="nirviReportUnexpectedCaseInternalError.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/nirviReportUnexpectedCaseInternalError.vi"/>
			<Item Name="niLvFpgaErrorClusterFromErrorCode.vi" Type="VI" URL="/&lt;vilib&gt;/rvi/errors/niLvFpgaErrorClusterFromErrorCode.vi"/>
			<Item Name="niLvFPGAUserSpecifiedEmulationVISupport.ctl" Type="VI" URL="/&lt;vilib&gt;/rvi/customViForEmulation/niLvFPGAUserSpecifiedEmulationVISupport.ctl"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
