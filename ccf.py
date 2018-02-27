

y1=y1-np.mean(y1)
y2=y2-np.mean(y2)

siy1=len(y1);
siy2=len(y2);

pad=65536;
number=siy1;

data1=np.zeros(pad);
data2=np.zeros(pad);

data1[:siy2]=y1;
data2[:siy2]=y2;

nn=np.floor(number/2);
findgen1=np.arange(0,nn)
findgen2=np.arange(0,nn);
norm1=number-nn+findgen1;
norm2=number-findgen2;

corr=np.zeros(2*nn);

fft1=np.fft.fft(data1);
fft2=np.fft.fft(data2);

pwrspc=np.conjugate(fft1)*fft2;

pwrspc=fft1*np.conjugate(fft2);

ztmp=np.real(np.fft.ifft(pwrspc));
norm=(np.sqrt(np.mean(y1*y1)*np.mean(y2*y2)))

corr[:nn]=ztmp[(pad-nn):(pad)]/norm/norm1;
corr[nn:(2*nn)]=ztmp[:nn]/norm/norm2;

tau=(findgen1+1)/fs;

tau2=np.flipud(-tau)
tau2=np.append(tau2,0)
tau2=np.append(tau2,tau)

return tau,corr