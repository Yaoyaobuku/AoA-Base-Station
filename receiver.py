import asyncio
import numpy as np
# from pylab import *
import matplotlib.pyplot as plt
from rtlsdr import *
# import threading
import time
import atexit
import threading

data={}

async def receive(sdr,i):
    async for samples in sdr.stream():
        # print("samples ("+str(samples.size)+"): "+str(samples))
        # print(str(time.time()))
        # seconds = time.time()

        # use matplotlib to estimate and plot the PSD
        # psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
        # xlabel('Frequency (MHz)')
        # ylabel('Relative power (dB)')
        #
        # show()

        global data

        if i not in data:
            data[i] = {}

        if 'counter' in data[i]:
            data[i]['counter'] += 1
        else:
            data[i]['counter'] = 0
        # data[i]['samples'] = samples
        if 'samples' in data[i]:
            data[i]['samples'] = np.concatenate([ samples, data[i]['samples'] ])
        else:
            data[i]['samples'] = samples
        print(len(samples))

        # y = np.fft.fft(samples)
        # x = np.linspace(0, 1, samples.size)
        # # freq = np.fft.fftfreq(samples.shape[-1])
        # plt.plot(x, y)
        # plt.show()

        # do something with samples
        # ...

    # to stop receive:
    await sdr.stop()

    # done
    sdr.close()

async def read_stream(filename,samples,i):
    print('enter read stream loop')
    while True:
        read(filename,samples,i)
        await asyncio.sleep(0.5)

def read(filename,samples,i):
    global data
    if i not in data:
        data[i] = {}
    if 'counter' in data[i]:
        data[i]['counter'] += 1
    else:
        # data[i]['counter'] = 100
        data[i]['counter'] = 25
    # data[i]['samples'] = np.fromfile(open(filename), dtype=np.complex64)
    if 'samples' in data[i]:
        data[i]['samples'] = np.concatenate([ np.fromfile(open(filename), dtype=np.complex64)[samples*data[i]['counter']:samples*(data[i]['counter']+1)], data[i]['samples'] ])
    else:
        data[i]['samples'] = np.fromfile(open(filename), dtype=np.complex64)[samples*data[i]['counter']:samples*(data[i]['counter']+1)]
    print('size: '+str(data[i]['samples'].size))

def data_available(n):
    for i in range(0,8):
        if i in data:
            if 'samples' in data[i]:
                if not data[i]['samples'].size >= n:
                    return False
            else:
                return False
        else:
            return False
    return True

def pop(n):
    for i in range(0,8):
        data[i]['samples'] = data[i]['samples'][n:]

async def plotloop( sample_rate, center_freq, i ):
    plt.ion()
    plt.show()
    while True:
        if data_available(1024*128):
            plt.clf()
            # print(data)
            for i in range(0,8):
                if i in data:
                    print('i: '+str(i)+' | counter: '+str(data[i]['counter']))
                    if data[i]['samples'].any() != None:
                        # y = np.fft.fft(data)
                        # x = np.linspace(0, 1, data.size)
                        # print(str(np.fft.fftfreq(samples.shape[-1])))
                        # plt.plot(x, y)

                        # ------------------------------
                        # FFT Plot
                        # ------------------------------
                        # ax1=plt.subplot(3, 1, 1)
                        # fft = np.fft.fft(data[i]['samples'][:1024*128])
                        # freq = (np.fft.fftfreq(len(fft),1/sample_rate)+center_freq)/1e6
                        # plt.plot(freq, 10*np.log10(fft), alpha=0.5)
                        # plt.title('FFT of Signals')
                        # plt.xlabel('Frequency (MHz)')
                        # plt.ylabel('Relative power (dB)')

                        # ------------------------------
                        # Time Domain Plot
                        # ------------------------------
                        # ax1=plt.subplot(3, 2, 3)
                        # plt.plot(range(0,len(data[i]['samples'][:1024*128])),data[i]['samples'][:1024*128], alpha=0.5)
                        # plt.title('Time Domain of Signals')
                        # plt.xlabel('Sample')
                        # plt.ylabel('Amplitude')

                        # ------------------------------
                        # Corrolation Plot + Shift
                        # ------------------------------
                        # if i >= 0:
                        if 'shift' not in data[i]:
                            if 0 in data:
                                ax1=plt.subplot(3, 2, 4)
                                fft_0 = np.fft.fft(data[0]['samples'][:1024*128])
                                fft_i = np.fft.fft(data[i]['samples'][:1024*128])
                                # fft_0 = np.fft.fft(data[0]['samples']*np.bartlett(data[0]['samples'].size))
                                # fft_i = np.fft.fft(data[i]['samples']*np.bartlett(data[i]['samples'].size))
                                corrolation = np.fft.ifft( fft_0 * np.conjugate(fft_i) )
                                # corrolation = np.correlate(data[0]['samples'],data[i]['samples'],'full')
                                # freq = np.fft.fftfreq(len(corrolation),1)
                                # array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])
                                x_p = range(0,int(corrolation.size/2))
                                x_n = range(-int(corrolation.size/2),0)
                                x = np.concatenate((x_p, x_n), axis=None)
                                # x = range(-int(corrolation.size/2),int(corrolation.size/2))
                                plt.plot(x, corrolation, alpha=0.5)
                                plt.title('Corrolation of Signals')
                                plt.xlabel('Samples Shifted')
                                plt.ylabel('Corrolation')
                                # if 'shift' not in data[i]:
                                print('Maximum is '+str(np.amax(corrolation))+' at position '+str(np.argmax(corrolation)))
                                data[i]['shift'] = np.argmax(corrolation)-data[i]['samples'].size
                                print('shift: '+str(data[i]['shift']))

                        if 'shift' in data[i]:
                            shifted_data = np.roll(data[i]['samples'][:1024*128], data[i]['shift'])
                            # else:
                            #     shifted_data = data[i]['samples']

                            # ------------------------------
                            # Time Domain Plot
                            # ------------------------------
                            # ax1=plt.subplot(3, 2, 5)
                            # plt.plot(range(0,len(shifted_data)),shifted_data, alpha=0.5)
                            # plt.title('Time Domain of Shifted Signals')
                            # plt.xlabel('Sample')
                            # plt.ylabel('Amplitude')

                            # ------------------------------
                            # Corrolation Plot
                            # ------------------------------
                            if i >= 0:
                                if 0 in data:
                                    ax1=plt.subplot(3, 2, 6)
                                    fft_0 = np.fft.fft(data[0]['samples'][:1024*128])
                                    fft_i = np.fft.fft(shifted_data)
                                    corrolation = np.fft.ifft( fft_0 * np.conjugate(fft_i) )
                                    # corrolation = np.correlate(data[0]['samples'],shifted_data,'full')
                                    # freq = (np.fft.fftfreq(len(corrolation),1/sample_rate)+center_freq)/1e6
                                    plt.plot(x, corrolation, alpha=0.5)
                                    plt.title('Corrolation of Shifted Signals')
                                    plt.xlabel('Samples Shifted')
                                    plt.ylabel('Corrolation')
                                    print('Maximum is '+str(np.amax(corrolation))+' at position '+str(np.argmax(corrolation)))

                        # ------------------------------
                        # Power Spectral Density Plot
                        # ------------------------------
                        # ax1=plt.subplot(2, 1, 2)
                        # plt.psd(data[i]['samples'], NFFT=1024, Fs=sample_rate/1e6, Fc=center_freq/1e6)
                        # plt.xlabel('Frequency (MHz)')
                        # plt.ylabel('Power Spectral Density (dB)')

                        # ------------------------------
                        # Update Plot
                        # ------------------------------
                        plt.tight_layout()
                        plt.draw()
                        plt.pause(0.001)
                    else:
                        print("No Data")
            pop(1024*64)
        else:
            print("Not Enough Data")
        # wait = input("PRESS ENTER TO CONTINUE.")
        await asyncio.sleep(0.2)
        # time.sleep(0.5)



def disconnect():
    global loop
    loop.close()
    global sdrs
    for sdr in sdrs:
        sdr.close()
atexit.register(disconnect)

# def data_acquisition_loop():

# sdr = RtlSdr()
# sdr = RtlSdr(serial_number='00000001')

tasks=[]

# ------------------------------
# Start SDRs
# ------------------------------
# sdrs=[]
# for i in range(0,8):
# # for i in range(0,1):
#     serial=str(i+1).zfill(8)
#     # serial="77771111153705700"
#     print(serial)
#     sdr = RtlSdr(serial_number=serial)
#     sdr.sample_rate = 1000000 #2048000
#     sdr.center_freq = 868e6
#     sdr.gain = 20.7
#     sdrs.append(sdr)
#     tasks.append(receive(sdr,i))
# tasks.append(plotloop(sdr.sample_rate,sdr.center_freq,1))

# ------------------------------
# File Stream
# ------------------------------
for i in range(0,8):
    # tasks.append(read('./data/rtl'+str(i+1),1024*32,i))
    tasks.append(read_stream('../data/rtl'+str(i+1),1024*128,i))
tasks.append(plotloop(2048000,868e6,1))



loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(*tasks))
# loops[i].run_until_complete(receive(sdr))
loop.close()


# threading.Thread(target=data_acquisition_loop, args=[]).start()
# plotloop(2048000, 868e6, 1);
