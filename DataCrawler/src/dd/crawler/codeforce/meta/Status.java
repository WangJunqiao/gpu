package dd.crawler.codeforce.meta;

public class Status {
	private String submissionId;
	private String contestId;
	private String timestamp;
	private String coderId;
	private String language;
	private String result;
	private String runTime; // ms
	private String runMemory; //KB
	
	
	public String toString() {
		return submissionId + " " + contestId + " " + timestamp + " " + coderId + " " + language + " " + runTime;
	}
	
	
	public String getSubmissionId() {
		return submissionId;
	}
	public void setSubmissionId(String submissionId) {
		this.submissionId = submissionId;
	}
	public String getContestId() {
		return contestId;
	}
	public void setContestId(String contestId) {
		this.contestId = contestId;
	}
	public String getTimestamp() {
		return timestamp;
	}
	public void setTimestamp(String timestamp) {
		this.timestamp = timestamp;
	}
	public String getCoderId() {
		return coderId;
	}
	public void setCoderId(String coderId) {
		this.coderId = coderId;
	}
	public String getLanguage() {
		return language;
	}
	public void setLanguage(String language) {
		this.language = language;
	}
	public String getResult() {
		return result;
	}
	public void setResult(String result) {
		this.result = result;
	}
	public String getRunTime() {
		return runTime;
	}
	public void setRunTime(String runTime) {
		this.runTime = runTime;
	}
	public String getRunMemory() {
		return runMemory;
	}
	public void setRunMemory(String runMemory) {
		this.runMemory = runMemory;
	}

}
