#!/usr/bin/env python

import os,sys
import ROOT
import math
import argparse

from ROOT import TFile, gDirectory
from ROOT import TH1F
from ROOT import *

def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True)
    return args.parse_args()



def fitplot():
    args = _get_args()
    hist=[]
    outputfiletable=open("mean.txt","w")
    myfile = TFile(args.input)
    mychain = gDirectory.Get('NNinput')

    mychain.Draw("NN_matrix0>>hist0")
    outputfiletable.write(str(hist0.GetMean())+" ")
    mychain.Draw("NN_matrix1>>hist1")
    outputfiletable.write(str(hist1.GetMean())+" ")
    mychain.Draw("NN_matrix2>>hist2")
    outputfiletable.write(str(hist2.GetMean())+" ")
    mychain.Draw("NN_matrix3>>hist3")
    outputfiletable.write(str(hist3.GetMean())+" ")
    mychain.Draw("NN_matrix4>>hist4")
    outputfiletable.write(str(hist4.GetMean())+" ")
    mychain.Draw("NN_matrix5>>hist5")
    outputfiletable.write(str(hist5.GetMean())+" ")
    mychain.Draw("NN_matrix6>>hist6")
    outputfiletable.write(str(hist6.GetMean())+" ")
    mychain.Draw("NN_matrix7>>hist7")
    outputfiletable.write(str(hist7.GetMean())+" ")
    mychain.Draw("NN_matrix8>>hist8")
    outputfiletable.write(str(hist8.GetMean())+" ")
    mychain.Draw("NN_matrix9>>hist9")
    outputfiletable.write(str(hist9.GetMean())+" ")
    mychain.Draw("NN_matrix10>>hist10")
    outputfiletable.write(str(hist10.GetMean())+" ")
    mychain.Draw("NN_matrix11>>hist11")
    outputfiletable.write(str(hist11.GetMean())+" ")
    mychain.Draw("NN_matrix12>>hist12")
    outputfiletable.write(str(hist12.GetMean())+" ")
    mychain.Draw("NN_matrix13>>hist13")
    outputfiletable.write(str(hist13.GetMean())+" ")
    mychain.Draw("NN_matrix14>>hist14")
    outputfiletable.write(str(hist14.GetMean())+" ")
    mychain.Draw("NN_matrix15>>hist15")
    outputfiletable.write(str(hist15.GetMean())+" ")
    mychain.Draw("NN_matrix16>>hist16")
    outputfiletable.write(str(hist16.GetMean())+" ")
    mychain.Draw("NN_matrix17>>hist17")
    outputfiletable.write(str(hist17.GetMean())+" ")
    mychain.Draw("NN_matrix18>>hist18")
    outputfiletable.write(str(hist18.GetMean())+" ")
    mychain.Draw("NN_matrix19>>hist19")
    outputfiletable.write(str(hist19.GetMean())+" ")
    mychain.Draw("NN_matrix20>>hist20")
    outputfiletable.write(str(hist20.GetMean())+" ")
    mychain.Draw("NN_matrix21>>hist21")
    outputfiletable.write(str(hist21.GetMean())+" ")
    mychain.Draw("NN_matrix22>>hist22")
    outputfiletable.write(str(hist22.GetMean())+" ")
    mychain.Draw("NN_matrix23>>hist23")
    outputfiletable.write(str(hist23.GetMean())+" ")
    mychain.Draw("NN_matrix24>>hist24")
    outputfiletable.write(str(hist24.GetMean())+" ")
    mychain.Draw("NN_matrix25>>hist25")
    outputfiletable.write(str(hist25.GetMean())+" ")
    mychain.Draw("NN_matrix26>>hist26")
    outputfiletable.write(str(hist26.GetMean())+" ")
    mychain.Draw("NN_matrix27>>hist27")
    outputfiletable.write(str(hist27.GetMean())+" ")
    mychain.Draw("NN_matrix28>>hist28")
    outputfiletable.write(str(hist28.GetMean())+" ")
    mychain.Draw("NN_matrix29>>hist29")
    outputfiletable.write(str(hist29.GetMean())+" ")
    mychain.Draw("NN_matrix30>>hist30")
    outputfiletable.write(str(hist30.GetMean())+" ")
    mychain.Draw("NN_matrix31>>hist31")
    outputfiletable.write(str(hist31.GetMean())+" ")
    mychain.Draw("NN_matrix32>>hist32")
    outputfiletable.write(str(hist32.GetMean())+" ")
    mychain.Draw("NN_matrix33>>hist33")
    outputfiletable.write(str(hist33.GetMean())+" ")
    mychain.Draw("NN_matrix34>>hist34")
    outputfiletable.write(str(hist34.GetMean())+" ")
    mychain.Draw("NN_matrix35>>hist35")
    outputfiletable.write(str(hist35.GetMean())+" ")
    mychain.Draw("NN_matrix36>>hist36")
    outputfiletable.write(str(hist36.GetMean())+" ")
    mychain.Draw("NN_matrix37>>hist37")
    outputfiletable.write(str(hist37.GetMean())+" ")
    mychain.Draw("NN_matrix38>>hist38")
    outputfiletable.write(str(hist38.GetMean())+" ")
    mychain.Draw("NN_matrix39>>hist39")
    outputfiletable.write(str(hist39.GetMean())+" ")
    mychain.Draw("NN_matrix40>>hist40")
    outputfiletable.write(str(hist40.GetMean())+" ")
    mychain.Draw("NN_matrix41>>hist41")
    outputfiletable.write(str(hist41.GetMean())+" ")
    mychain.Draw("NN_matrix42>>hist42")
    outputfiletable.write(str(hist42.GetMean())+" ")
    mychain.Draw("NN_matrix43>>hist43")
    outputfiletable.write(str(hist43.GetMean())+" ")
    mychain.Draw("NN_matrix44>>hist44")
    outputfiletable.write(str(hist44.GetMean())+" ")
    mychain.Draw("NN_matrix45>>hist45")
    outputfiletable.write(str(hist45.GetMean())+" ")
    mychain.Draw("NN_matrix46>>hist46")
    outputfiletable.write(str(hist46.GetMean())+" ")
    mychain.Draw("NN_matrix47>>hist47")
    outputfiletable.write(str(hist47.GetMean())+" ")
    mychain.Draw("NN_matrix48>>hist48")
    outputfiletable.write(str(hist48.GetMean())+" ")


    mychain.Draw("NN_pitches0>>hist49")
    outputfiletable.write(str(hist49.GetMean())+" ") 
    mychain.Draw("NN_pitches1>>hist50")
    outputfiletable.write(str(hist50.GetMean())+" ")
    mychain.Draw("NN_pitches2>>hist51")
    outputfiletable.write(str(hist51.GetMean())+" ")
    mychain.Draw("NN_pitches3>>hist52")
    outputfiletable.write(str(hist52.GetMean())+" ")
    mychain.Draw("NN_pitches4>>hist53")
    outputfiletable.write(str(hist53.GetMean())+" ")
    mychain.Draw("NN_pitches5>>hist54")
    outputfiletable.write(str(hist54.GetMean())+" ")
    mychain.Draw("NN_pitches6>>hist55")
    outputfiletable.write(str(hist55.GetMean())+" ")

    mychain.Draw("NN_layer>>hist100")
    outputfiletable.write(str(hist100.GetMean())+" ")
    mychain.Draw("NN_barrelEC>>hist101")
    outputfiletable.write(str(hist101.GetMean())+" ")
    mychain.Draw("NN_phi>>hist102")
    outputfiletable.write(str(hist102.GetMean())+" ")
    mychain.Draw("NN_theta>>hist103")
    outputfiletable.write(str(hist103.GetMean())+"\n")
fitplot()