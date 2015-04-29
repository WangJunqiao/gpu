#include "DataSource.h"
#include <io.h>
#include <direct.h>
#include <assert.h>

bool DataSource::_browse_dir(string path) {
	if (_access(path.c_str(), 06) != 0) {
		cerr<<"error: directory does not exit!"<<endl;
		return false;
	}
	assert(path.length() >= 1);
	int len = path.length();
	if (path[len - 1] == '/' || (len>1 && path[len-1]=='\\' && path[len-2]=='\\')){
		//do nothing
	}
	else path += '/';

	len = path.length();
	if (chdir(path.c_str()) != 0) { //not a directory, take it as a file
		if (path[len - 1] == '/')path.resize(len - 1);
		else if (path[len - 1] == '\\')path.resize(len - 2);
		_doc_paths.push_back(path);
		return true;
	}

	_finddata_t *file_info = new _finddata_t;
	memset(file_info, 0x0, sizeof(file_info));
	intptr_t index = _findfirst("*", file_info); //get first file
	if (file_info->attrib & _A_SUBDIR) { //if it is a directory
		if(strcmp(file_info->name, ".") && strcmp(file_info->name, "..")) {
			string t_path = path + file_info->name + '/';
			_browse_dir(t_path); //go into the sub directory
		}
	}
	else {
		string t_path = path + file_info->name;
		_doc_paths.push_back(t_path);
	}
	while(_findnext(index, file_info) == 0) { 
		if ((file_info->attrib & _A_SUBDIR)) { 
			if(strcmp(file_info->name, ".") && strcmp(file_info->name, "..")) {
				string t_path = path + file_info->name + '/';
				_browse_dir(t_path); 
			}
		}
		else {
			string t_path = path + file_info->name;
			_doc_paths.push_back(t_path);
		}
	}
	return true;
}

bool DataSource::openSource(){
	return DataSource::_browse_dir(_path);
}

bool DataSource::hasNext() {
	return _file_index < _doc_paths.size();
}

string DataSource::getNextDocument() {
	//before get, you should see if it has next doc
	FILE *tfp = fopen(_doc_paths[_file_index].c_str(), "r");
	assert(tfp != NULL);
	string s;
	char buffer[1000000];
	while(fgets(buffer, 1000000, tfp)!=NULL)
		s += buffer;
	assert(feof(tfp) != 0);
	_file_index++;
	fclose(tfp);
	return s;
}

void DataSource::closeSource() {
	fclose(fp);
}

string DataSource::getDocName(const int &docid) {
	assert(docid <= _doc_paths.size() && docid > 0);
	string doc_name = _doc_paths[docid - 1];
	string result;
	for (int size = doc_name.size(), i = size - 1; i >= 0 
		&& doc_name[i] != '\\' && doc_name[i] != '/'; i--) {
		result += doc_name[i];
	}
	return result;
}

vector<string> DataSource::getDocNames() {
	return _doc_paths;
}

