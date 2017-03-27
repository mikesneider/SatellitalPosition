# coding=utf-8
#For Mike Puentes - Universidad Industrial de Santander
#Now I need to know the near 

__author__ = 'mikesneider@gmail.com'
import datetime
import ephem
import math
import os
import sys 
import time
import urllib2
import re
import numpy as np
import geocoder


float_formatter = lambda x: "%.2f" % x

class WGS84:
    """General parameters defined by the WGS84 system"""
    #Semimajor axis length (m)
    a = 6378137.0
    #Semiminor axis length (m)
    b = 6356752.3142
    #Ellipsoid flatness (unitless)
    f = (a - b) / a
    #Eccentricity (unitless)
    e = math.sqrt(f * (2 - f))
    #Speed of light (m/s)
    c = 299792458.
    #Relativistic constant
    F = -4.442807633e-10
    #Earth's universal gravitational constant
    mu = 3.986005e14
    #Earth rotation rate (rad/s)
    omega_ie = 7.2921151467e-5
    secsInWeek = 604800 
    secsInDay = 86400 
    gpsEpoch = (1980, 1, 6, 0, 0, 0)  # (year, month, day, hh, mm, ss)
    
    
def GetPositionELAZ(Time,Longitude,Latitude,constelation):
    '''
    - time: Time for look the satellite
    - longitude: observer longitude
    - latitude: observer latitude
    - constelation: 
                1 - just GPS 
                2 - GPS - GLONASS
                3 - GPS - GLONASS - GALILEO
                
    RETURN: (Tuble of lists...)
    Satellite Name, Satellite Elevation, Satellite Azimuth, Type Satellite (1:GPS, 2: GLO, 3:GAL)
    '''
    contain_GPS = 'GPS'
    contain_GLONASS = 'COSMOS'
    contain_GALILEO = 'GSAT'
    
    sat_alt, sat_az, sat_name, sat_type = [], [], [], []

    longlat = str(Longitude) + "," + str(Latitude)
    g = geocoder.elevation(longlat) #lon,lat
    
    observer = ephem.Observer()
    observer.long = Longitude
    observer.lat = Latitude
    observer.date = Time
    observer.elevation = float(str(g.meters))
    
    GPS_list = 'http://www.celestrak.com/NORAD/elements/gps-ops.txt'
    GLONASS_list = 'http://www.celestrak.com/NORAD/elements/glo-ops.txt'
    GALILEO_list = 'http://www.celestrak.com/NORAD/elements/galileo.txt'
    
    #'http://www.amsat.org/amsat/ftp/keps/current/nasabare.txt').readlines()
    GPS2_list = 'http://www.tle.info/data/gps-ops.txt'
    GLONASS2_list = 'http://www.tle.info/data/glo-ops.txt'
    
    if constelation == 1:
        list_TLEs = [GPS_list]
    if constelation == 2:
        list_TLEs = [GPS_list,GLONASS_list]
    if constelation == 3:
        list_TLEs = [GPS_list,GLONASS_list,GALILEO_list]
    
    for satlist in list_TLEs:
        tles = urllib2.urlopen(satlist).readlines() 
        tles = [item.strip() for item in tles]
        tles = [(tles[i],tles[i+1],tles[i+2]) for i in xrange(0,len(tles)-2,3)]
        
        for tle in tles:

            try:
                sat = ephem.readtle(tle[0], tle[1], tle[2])
                rt, ra, tt, ta, st, sa = observer.next_pass(sat)

                if rt is not None and st is not None:
                    sat.compute(observer)

                    if TimeNow >= ephem.localtime(st) and TimeNow <= ephem.localtime(rt):
                        Title = tle[0]
                        sat_alt.append(np.rad2deg(sat.alt))
                        sat_az.append(np.rad2deg(sat.az))

                        text2 = Title.rsplit(')', 1)[0]
                        NamePRN = text2.rsplit('(', 1)[1]
                        sat_name.append(NamePRN)

                        if contain_GPS in Title:
                            sat_type.append(1)
                        elif contain_GLONASS in Title:
                            sat_type.append(2)
                        elif contain_GALILEO in Title:
                            sat_type.append(3)

                            #I wanna Try just with the PRN 1
                            #if NamePRN = 'PRN 01':
                            #print NamePRN
                            #print "Elev:" , np.rad2deg(sat.alt), "Azimuth: ",np.rad2deg(sat.az) #sat.alt / ephem.degree
                            #print "original rise time: ",rt
                            #print 'rise time: ', ephem.localtime(rt)
                            #print 'set time: ',  ephem.localtime(st) 
                            #print
                            #print 'Time until set: ',ephem.localtime(st) - TimeNow
                            #print 'Time until rise: ', ephem.localtime(rt) - TimeNow
                            #print 'Longitude: ', sat.sublong
                            #print 'Latitude: ',sat.sublat
                            #timeuntilrise = ephem.localtime(rt)-TimeNow
                            #HowMany += 1
                            #minutesaway = timeuntilrise.seconds/60.0
                            #print "Minutes Away: ",minutesaway

            except ValueError as e:
                print e
    return sat_name, sat_alt, sat_az, sat_type

def to_skyplot(elevation, azimut):
    '''
    elevation: in grades
    azimut: in grades
    '''
    e = np.array(elevation)
    a = np.array(azimut)
    sx = ((90-e)/90) * np.sin (a * (np.pi/180))
    sy = ((90-e)/90) * np.cos (a * (np.pi/180))
    return (sx,sy)

def GetDateTime(Date,Time):
    '''
    Date: String of date like (01/02/1987)
    Time: String of time like (13:34:56)
    '''
    from datetime import datetime
    NewTime = datetime.strptime(date +' '+ time, '%d/%m/%Y %H:%M:%S')
    return NewTime
def colorsyou(lista):
    colores = []
    for num in lista:
        if num == 1:
            colores.append('red')
        if num == 2:
            colores.append('blue')
        if num == 3:
            colores.append('yellow')
    return colores
    
