import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io import shapereader
import numpy as np
import netCDF4 as nc
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter

# 绘制中国地图
shp_path = r'G:\HZY\china-shp\china-ali\中华人民共和国.shp'   #改成自己的
filename = r'G:\data\shuiwenqixiang\caijian_tcrw.nc'   #改成自己的
depth = xr.open_dataset(filename)
var_data1 = nc.Dataset(filename)
var_data = var_data1.variables['tcrw'][:]
years_data = np.add.reduceat(var_data, np.arange(0, var_data.shape[0], 12), axis=0)
years_data[years_data< 0] = float('nan')

mask = np.isnan(years_data)
bb = np.ma.masked_array(years_data, mask=mask)

masked_data = np.ma.masked_array(years_data, mask=mask)
aa = np.mean(masked_data,axis=0)
from tqdm import  tqdm
import pymannkendall as mk
resz = np.zeros((190,245))
for i in tqdm(range(190)):
    for j in range(245):
        if bb[0,0,i,j] is np.ma.masked:
            continue
        else:
            res = mk.original_test(bb[:,0,i,j],alpha=0.05)#alpha默认为0.05
            resz[i,j] = res.z
resz[resz== 0] = float('nan')
plt.imshow(resz)
plt.colorbar()
plt.show()
#创建一个xrarray

lons = depth.longitude.data
lats = depth.latitude.data
expver = depth.expver.data
time = depth.time.data[0:85]
temp = xr.DataArray(aa, coords=[expver,lats,lons], dims=['expver','latitude','longitude'])
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
reader = shapereader.Reader(shp_path)
for record in reader.records():
    ax.add_geometries([record.geometry], ccrs.PlateCarree(), facecolor='none')
ax.set_extent([73, 138, 14, 54])
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True) )
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks(np.arange(73,138,15), crs=proj)
ax.set_yticks(np.arange(14,54,15), crs=proj)
# 绘制南海子图
sub_ax = fig.add_axes([0.556, 0.122, 0.25, 0.25],
                      projection=ccrs.PlateCarree())
for record in reader.records():
    sub_ax.add_geometries([record.geometry], ccrs.PlateCarree(), facecolor='none')
sub_ax.set_extent([105, 125, 3, 25], crs=ccrs.PlateCarree())
sub_ax.contourf(temp.longitude[128:209], temp.latitude[0:157], temp[0,0:157,128:209],cmap='RdYlBu',
    extend='both', alpha=0.8)
#im2 = ax.imshow(ds.longitude, ds.latitude, ds.t2m[0,:,:])
im = ax.contourf(
    temp.longitude, temp.latitude, temp[0,:,:],cmap='RdYlBu',
    extend='both', alpha=0.8)
cbar = fig.colorbar(im)
cbar.ax.set_title('Units:K')
fig.suptitle('Distribution of annual mean temperature in China',fontsize=20, x=0.5, y=0.95)
plt.show()
plt.savefig(r'G:\data\shuiwenqixiang\t2m_avg.jpg')
